import os
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from string import ascii_letters, digits
from typing import Optional, Tuple, Union, Any, Dict
from urllib.parse import urljoin

import driverlessai
import h2osteam
import httpx
from h2o_wave import Q
from h2osteam.clients import DriverlessClient, H2oKubernetesClient
from h2osteam.clients.driverless import DriverlessInstance
from h2osteam.clients.h2ok8s.h2ok8s import H2oKubernetesCluster
from h2osteam.backend import SteamConnection
from loguru import logger


def in_cloud():
    return os.getenv('KUBERNETES_PORT') is not None


def init_steam_lib(q: Q):
    q.app.env_vars = dict()
    q.app.users = dict()
    q.app.callbacks = {}


def clear_steam_cards(q: Q):
    for card in ['engine_dashboard_header', 'engine_dashboard']:
        del q.page[card]


class AppUser:
    def __init__(self, user_id, email):
        self.user_id = user_id
        self.email = email
        self._set_name()

        # User's Steam Engines
        self.dai_engines = {}
        self.h2o3_engines = {}


    def _set_name(self):
        names = self.email.split('@')[0].split('.')
        if len(names) > 1:
            self.first, *_, self.last = names
        elif names:
            self.first = names[0]
            self.last = ''
        self.name = f'{self.first} {self.last}'.strip().title()



@dataclass
class H2oEngineConfig:
    name: str
    version: str
    node_count: int
    cpu_count: int
    memory_gb: int
    max_idle_h: int
    max_uptime_h: int
    timeout_s: int = 600
    profile_name: Optional[Union[str, None]] = 'default-h2o-kubernetes'
    dataset_size_gb: Optional[Union[float, None]] = None
    dataset_dimension: Optional[Union[Tuple[int, int], None]] = None
    gpu_count: int = 0

    def __post_init__(self):
        self.timeout_s = max(self.timeout_s, 600)


@dataclass
class DAIEngineConfig:
    name: str
    version: str
    cpu_count: int
    cpu_count: int
    gpu_count: int
    memory_gb: int
    storage_gb: int
    max_idle_h: int
    max_uptime_h: int
    timeout_s: int = 600
    profile_name: Optional[Union[str, None]] = 'default-driverless-kubernetes'
    sync: bool = True

    def __post_init__(self):
        self.timeout_s = max(self.timeout_s, 600)


@dataclass
class SteamCredentials:
    access_token: Optional[Union[str, None]] = None
    steam_url: Optional[Union[str, None]] = None
    steam_api_token: Optional[Union[str, None]] = None
    valid: bool = field(init=False, default=False)

    def __post_init__(self):
        if in_cloud():
            # Verify access_token
            if self.access_token is None:
                logger.error('Need to provide access_token when running in cloud !')

            # Verify steam_url
            if self.steam_url is None:
                logger.debug('No steam_url provided, getting it from STEAM_URL')
                self.steam_url = os.environ.get('STEAM_URL')
            if self.steam_url is None:
                logger.error(
                    'Unable to obtain STEAM_URL from HAC environment.'
                    ' Unable to connect to Steam!'
                )

            # if both access_token and steam_url are available, mark as valid
            if self.access_token and self.steam_url:
                self.valid = True
                self.steam_api_token = None
        else:
            # Verify steam_api_token
            if self.steam_api_token is None:
                logger.debug(
                    'No steam_api_token provided, getting it from STEAM_API_TOKEN'
                )
                self.steam_api_token = os.environ.get('STEAM_API_TOKEN')
            if self.steam_api_token is None:
                logger.error(
                    'Unable to obtain STEAM_API_TOKEN from local environment.'
                    ' Unable to connect to Steam!'
                )

            # Verify steam_url
            if self.steam_url is None:
                logger.debug('No steam_url provided, getting it from DEV_STEAM_URL')
                self.steam_url = os.environ.get('DEV_STEAM_URL')
            if self.steam_url is None:
                logger.error('Need to provide steam_url if not running in HAC')

            # if both steam_api_token and steam_url are available, mark as valid
            if self.steam_api_token and self.steam_url:
                self.valid = True
                self.access_token = None


@dataclass
class DAIEngineDetails:
    id: int
    profile_name: str
    name: str
    status: str
    target_status: str
    version: str
    instance_type: str
    master_id: int
    cpu_count: int
    gpu_count: int
    memory_gb: int
    storage_gb: int
    max_idle_seconds: int
    max_uptime_seconds: int
    timeout_seconds: int
    password: str
    created_at: datetime
    started_at: datetime
    created_by: str
    current_uptime_seconds: int
    current_idle_seconds: int
    # backend_type:
    # address: http://10.3.191.212:12345
    # authentication: oidc
    # pod_latest_event: None
    # service_latest_event: None
    # pvc_latest_event: None
    # config_toml: None

    @classmethod
    def load(cls, details):
        required_fields = [f.name for f in fields(cls)]
        for k in list(details.keys()):
            if k not in required_fields:
                del details[k]
        details['created_at'] = datetime.utcfromtimestamp(details['created_at'])
        details['started_at'] = datetime.utcfromtimestamp(details['started_at'])
        return cls(**details)  # type: ignore


@dataclass
class H2O3EngineDetails:
    name: str
    status: str
    id: int
    profile_name: str
    target_status: str
    version: str
    node_count: int
    cpu_count: int
    gpu_count: int
    memory_gb: int
    max_idle_h: int
    max_uptime_h: int
    context_path: str
    created_at: datetime
    created_by: str
    timeout_seconds: int = 0
    current_uptime_millis: int = 0
    current_idle_millis: int = 0

    @classmethod
    def load(cls, h2o3_cluster: H2oKubernetesCluster):
        try:
            details = h2o3_cluster._api.get_h2o_kubernetes_cluster(h2o3_cluster.name)
        except Exception as e:
            logger.warning(
                f'Unable to get h2o3 kubernetes cluster using internal api for {h2o3_cluster.name}\n{e}'
            )
        else:
            logger.debug(
                f'Reading details from h2o3 kubernetes cluster using internal api for {h2o3_cluster.name}'
            )
            details['created_at'] = datetime.utcfromtimestamp(details['created_at'])
            details['max_idle_h'] = details['max_idle_hours']
            details['max_uptime_h'] = details['max_uptime_hours']
            del details['max_idle_hours']
            del details['max_uptime_hours']
            return cls(**details)  # type: ignore
        details = dict(
            id=h2o3_cluster.id,
            name=h2o3_cluster.name,
            profile_name=h2o3_cluster.profile_name,
            status=h2o3_cluster.status,
            target_status=h2o3_cluster.target_status,
            version=h2o3_cluster.version,
            node_count=h2o3_cluster.node_count,
            cpu_count=h2o3_cluster.cpu_count,
            gpu_count=h2o3_cluster.gpu_count,
            memory_gb=h2o3_cluster.memory_gb,
            max_idle_h=h2o3_cluster.max_idle_h,
            max_uptime_h=h2o3_cluster.max_uptime_h,
            context_path=h2o3_cluster.context_path,
            created_at=datetime.strptime(h2o3_cluster.created_at, "%Y-%m-%dT%H:%M:%S"),
            created_by=h2o3_cluster.created_by,
        )
        return cls(**details)  # type: ignore


def __create_steam_connection_hac(
    steam_url, access_token
) -> Union[None, SteamConnection]:
    logger.debug('Trying to log into Steam while in HAC')
    try:
        steam_connection = h2osteam.login(
            url=steam_url,
            access_token=access_token,
            verify_ssl=not steam_url.startswith('http://'),
        )
    except Exception as e:
        logger.error(f'Unable to Connect to Steam when running in HAC\n{e}')
        return None
    else:
        return steam_connection


def __create_steam_connection_local(
    steam_url, steam_api_token
) -> Union[None, SteamConnection]:
    logger.debug('Trying to log into Steam while NOT in HAC')
    try:
        steam_connection = h2osteam.login(url=steam_url, password=steam_api_token)
    except Exception as e:
        logger.error(f'Unable to Connect to Steam when running outside HAC\n{e}')
        return None
    else:
        return steam_connection


def create_steam_connection(
    steam_url, access_token=None, steam_api_token=None
) -> Union[None, SteamConnection]:
    connection = None
    if access_token and not steam_api_token:
        connection = __create_steam_connection_hac(
            steam_url=steam_url,
            access_token=access_token,
        )
    elif steam_api_token and not access_token:
        connection = __create_steam_connection_local(
            steam_url=steam_url,
            steam_api_token=steam_api_token,
        )
    return connection


class DAISteamEngine:
    def __init__(
        self,
        steam_connection: SteamConnection,
        config: DAIEngineConfig = None,
        dai_engine_details: Dict[str, Any] = None,
    ):
        self.steam_connection = steam_connection

        self.name = None
        self.openid_login_url = None
        self.details = None
        self.created = False
        self.terminated = False
        self.status = None
        self.last_updated = datetime.now()

        if self.steam_connection is not None:
            if config is not None:
                dai_instance = self._create(config)
                if dai_instance is not None:
                    updated = self.update(
                        dai_instance=dai_instance,
                    )
                    if updated:
                        self.created = True
            elif dai_engine_details:
                self.details = DAIEngineDetails.load(dai_engine_details)
                self.name = self.details.name
                self.created = True
                self.status = self.details.status
                self._set_openid_login_url()
                self.last_updated = datetime.now()

    def _create(self, config: DAIEngineConfig) -> Union[DriverlessInstance, None]:
        if not self.steam_connection:
            return None
        try:
            # TODO: make this sync=False
            dai_instance: DriverlessInstance = DriverlessClient(
                self.steam_connection
            ).launch_instance(**asdict(config))
        except Exception as e:
            logger.error(f'Unable to create new DriverlessAI instance in Steam.\n{e}')
            return None
        else:
            self.name = config.name
            self.openid_login_url = dai_instance.openid_login_url()
            return dai_instance

    def _set_openid_login_url(self):
        dai_instance = self._get_instance()
        if dai_instance is not None:
            self.openid_login_url = dai_instance.openid_login_url()

    def _get_instance(self) -> Union[DriverlessInstance, None]:
        if not self.steam_connection:
            return None
        try:
            dai_instance: DriverlessInstance = DriverlessClient(
                self.steam_connection
            ).get_instance(name=self.details.name)
        except Exception as e:
            logger.error(
                f'Unable to connect to newly created DriverlessAI instance.\n{e}'
            )
            return None
        else:
            return dai_instance

    def update(self, dai_instance=None) -> bool:
        if dai_instance is None:
            dai_instance = self._get_instance()
        if dai_instance is None:
            return False
        try:
            details = dai_instance.details()
        except Exception as e:
            logger.error(f'Unable to update details\n{e}')
            return False
        else:
            required_fields = [f.name for f in fields(DAIEngineDetails)]
            for k in list(details.keys()):
                if k not in required_fields:
                    del details[k]
            details['created_at'] = datetime.utcfromtimestamp(details['created_at'])
            details['started_at'] = datetime.utcfromtimestamp(details['started_at'])

            self.details = DAIEngineDetails(**details)  # type: ignore
            self.last_updated = datetime.now()
            self.status = self.details.status
            return True

    def _wait(self, target_status, timeout=600) -> bool:
        start_time = datetime.now()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        if not self.update():
            return False
        while self.status != target_status and elapsed_time <= timeout:
            time.sleep(5)
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if not self.update():
                return False
            logger.debug(f'elapsed: {elapsed_time} seconds ...')
            logger.debug(f'{self.name} - status: {self.status}')

        if self.status == target_status:
            logger.info(
                f'Driverless AI Engine {self.name} is now in {target_status} state after {elapsed_time} seconds.'
            )
            return True
        logger.error(
            f'Driverless AI Engine {self.name} is NOT in {target_status} state after {elapsed_time} seconds.'
        )
        return False

    def start(self, timeout=600) -> bool:
        if not self.update():
            return False
        dai_instance = self._get_instance()
        if dai_instance is None:
            return False
        try:
            dai_instance.start(sync=False)
        except Exception as e:
            logger.error(f'Unable to start DAI Engine\n{e}')
            return False
        else:
            started = self._wait(target_status='running', timeout=timeout)
            return started

    def stop(self, timeout=600) -> bool:
        if not self.update():
            return False
        dai_instance = self._get_instance()
        if dai_instance is None:
            return False
        try:
            dai_instance.stop(sync=False)
        except Exception as e:
            logger.error(f'Unable to stop DAI Engine\n{e}')
            return False
        else:
            stopped = self._wait(target_status='stopped', timeout=timeout)
            return stopped

    def _wait_for_termination(self, timeout=600) -> bool:
        start_time = datetime.now()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        if not self.update():
            logger.info(
                f'Driverless AI Engine {self.name} is terminated after {elapsed_time} seconds.'
            )
            return True
        while elapsed_time <= timeout:
            time.sleep(5)
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if not self.update():
                logger.info(
                    f'Driverless AI Engine {self.name} is terminated after {elapsed_time} seconds.'
                )
                return True
            logger.debug(f'elapsed: {elapsed_time} seconds ...')
            logger.debug(f'{self.name} - status: {self.status}')

        if self.update():
            logger.error(
                f'Driverless AI Engine {self.name} did not terminate after {elapsed_time} seconds.'
            )
            return False
        logger.info(
            f'Driverless AI Engine {self.name} is terminated after {elapsed_time} seconds.'
        )
        return True

    def terminate(self, timeout=600) -> bool:
        if not self.update():
            logger.warning(
                f'Driverless AI Engine {self.name} is unreachable or already terminated.'
            )
            return True
        dai_instance = self._get_instance()
        if dai_instance is None:
            self.terminated = True
            return True
        try:
            dai_instance.terminate(sync=False)
        except Exception as e:
            logger.error(f'Unable to terminate DAI Engine\n{e}')
            return False
        else:
            terminated = self._wait_for_termination(timeout=timeout)
            self.terminated = terminated
            return terminated

    def connect(self) -> Any:
        if not self.update():
            return None
        dai_instance = self._get_instance()
        if dai_instance is None:
            return None
        try:
            h2oai = dai_instance.connect()
        except Exception as e:
            logger.error(f'Unable to connect to DAI Engine\n{e}')
            return None
        else:
            logger.info(f'Successfully connected to {self.name}')
            return h2oai


class H2O3SteamEngine:
    def __init__(
        self,
        steam_connection: SteamConnection,
        timeout: int = 600,
        config: H2oEngineConfig = None,
        h2o3_cluster: Any = None,
    ):
        self.steam_connection = steam_connection

        self.name = None
        self.url = None
        self.details = None
        self.created = False
        self.terminated = False
        self.status = None
        self.last_updated = datetime.now()

        if self.steam_connection:
            if config is not None:
                h2o3_cluster = self._create(config, timeout=timeout)
                if h2o3_cluster is not None:
                    self._set_url()
                    self.created = True
            elif h2o3_cluster:
                self.details = H2O3EngineDetails.load(h2o3_cluster)
                self.name = self.details.name
                self.created = True
                self.status = self.details.status
                self._set_url()
                self.last_updated = datetime.now()

    def _create(
        self, config: H2oEngineConfig, timeout=600
    ) -> Union[H2oKubernetesCluster, None]:
        if not self.steam_connection:
            return None
        try:
            h2o3_cluster = H2oKubernetesClient(self.steam_connection).launch_cluster(
                **asdict(config)
            )
        except Exception as e:
            logger.error(f'Unable to create new H2O-3 cluster in Steam.\n{e}')
            return None
        else:
            logger.info(f'Created new H2O-3 Cluster {self.name}')
            self.name = h2o3_cluster.name
            self.status = h2o3_cluster.status
            is_running = self._wait(target_status='running', timeout=timeout)
            if is_running:
                return h2o3_cluster

    def _get_cluster(self) -> Union[H2oKubernetesCluster, None]:
        if not self.steam_connection:
            return None
        try:
            h2o3_cluster: H2oKubernetesCluster = H2oKubernetesClient(
                self.steam_connection
            ).get_cluster(self.name)
        except Exception as e:
            logger.error(f'Unable to get H2O-3 cluster.\n{e}')
            return None
        else:
            return h2o3_cluster

    def _set_url(self):
        steam_perfix = 'https://steam.demo.h2o.ai/proxy/'
        h2o_head = 'h2o-k8s/'
        h2o_tail = '/flow/index.html'
        self.url = steam_perfix + h2o_head + str(self.details.id) + h2o_tail

    def update(self) -> bool:
        h2o3_cluster = self._get_cluster()
        if h2o3_cluster is None:
            return False
        self.details = H2O3EngineDetails.load(h2o3_cluster)
        self.last_updated = datetime.now()
        self.status = self.details.status
        return True

    def _wait(self, target_status, timeout=600) -> bool:
        start_time = datetime.now()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        if not self.update():
            return False
        while self.status != target_status and elapsed_time <= timeout:
            time.sleep(5)
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if not self.update():
                return False
            logger.debug(f'elapsed: {elapsed_time} seconds ...')
            logger.debug(f'{self.name} - status: {self.status}')

        if self.status == target_status:
            logger.info(
                f'H2O-3 Steam Engine {self.name} is now in {target_status} state after {elapsed_time} seconds.'
            )
            return True
        logger.error(
            f'H2O-3 Steam Engine {self.name} is NOT in {target_status} state after {elapsed_time} seconds.'
        )
        return False

    def _wait_for_termination(self, timeout=600) -> bool:
        start_time = datetime.now()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        if not self.update():
            logger.info(
                f'H2O-3 Cluster {self.name} is terminated after {elapsed_time} seconds.'
            )
            return True
        while elapsed_time <= timeout:
            time.sleep(5)
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if not self.update():
                logger.info(
                    f'H2O-3 Cluster {self.name} is terminated after {elapsed_time} seconds.'
                )
                return True
            logger.debug(f'elapsed: {elapsed_time} seconds ...')
            logger.debug(f'{self.name} - status: {self.status}')

        if self.update():
            logger.error(
                f'H2O-3 Steam Engine {self.name} did not terminate after {elapsed_time} seconds.'
            )
            return False
        logger.info(
            f'H2O-3 Steam Engine {self.name} is terminated after {elapsed_time} seconds.'
        )
        return True

    def terminate(self, timeout=600):
        if not self.update():
            logger.warning(
                f'H2O-3 Steam Engine {self.name} is unreachable or already terminated.'
            )
        h2o3_cluster = self._get_cluster()
        if h2o3_cluster is None:
            self.terminated = True
            return True
        try:
            h2o3_cluster.terminate()
        except Exception as e:
            logger.error(f'Unable to terminate H2O-3 Steam Engine\n{e}')
            return False
        else:
            terminated = self._wait_for_termination(timeout=timeout)
            self.terminated = terminated
            return terminated


async def get_dai_localhost(dai_localhost) -> bool:
    dai_url = urljoin(dai_localhost, 'version')
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(dai_url)
    except Exception as e:
        logger.info('Error while trying to ping DAI at localhost!')
        return False
    else:
        if r.status_code == httpx.codes.OK:
            r_json = r.json()
            if r_json.get('webAppVersion'):
                return True
    return False


def connect_to_dai(dai_url, dai_user, dai_pw):
    try:
        _ = driverlessai.Client(
            address=dai_url,
            username=dai_user,
            password=dai_pw,
        )
    except Exception as e:
        logger.warning(f'Unable to connect to DAI {e}')
    else:
        return True
    return False


async def get_h2o3_localhost(h2o3_localhost) -> bool:
    h2o3_url = urljoin(h2o3_localhost, '/3/Cloud')
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(h2o3_url)
    except Exception as e:  # noqa: F541
        logger.info('Error while trying to ping h2o3 at localhost!')
        return False
    else:
        if r.status_code == httpx.codes.OK:
            r_json = r.json()
            if r_json.get('version') and r_json.get('cloud_healthy'):
                return True
    return False


async def get_steam_engines(q: Q):
    connected = False
    make_user_steam_connection(q)

    if q.user.steam_connection:
        connected = True
        # if there is no local dict for dai_engines, create one
        if q.user.user.dai_engines is None:
            q.user.user.dai_engines = {}

        # Get latest DriverlessAI instances in Steam
        dai_steam = {}
        for dai_ in h2osteam.api().get_driverless_instances():
            if dai_['created_by'] == q.user.user.email:
                dai_steam_engine = DAISteamEngine(
                    steam_connection=q.user.steam_connection, dai_engine_details=dai_
                )
                dai_steam[dai_['name']] = {
                    'type': 'steam',
                    'engine': dai_steam_engine,
                }

        # If an instance is in local dict but not in the latest update, remove it
        for k, v in list(q.user.user.dai_engines.items()):
            if v['type'] == 'steam' and k not in dai_steam:
                del q.user.user.dai_engines[k]

        # Update the local dict with latest Steam update
        q.user.user.dai_engines.update(dai_steam)

        # if there is no local dict for h2o3_engines, create one
        if q.user.user.h2o3_engines is None:
            q.user.user.h2o3_engines = {}

        h2o3_steam = {}
        for h2o3_ in H2oKubernetesClient().get_clusters():
            if h2o3_.created_by == q.user.user.email:
                h2o3_steam_engine = H2O3SteamEngine(
                    steam_connection=q.user.steam_connection, h2o3_cluster=h2o3_
                )
                h2o3_steam[h2o3_.name] = {
                    'type': 'steam',
                    'engine': h2o3_steam_engine,
                }

        for k, v in list(q.user.user.h2o3_engines.items()):
            if v['type'] == 'steam' and k not in h2o3_steam:
                del q.user.user.h2o3_engines[k]

        # Update the local dict with latest Steam update
        q.user.user.h2o3_engines.update(h2o3_steam)
        q.user.last_steam_refresh = datetime.now()

    # Check if DAI and H2O-3 are running on local machine
    if not in_cloud():
        if await get_dai_localhost('http://localhost:12345'):
            q.user.user.dai_engines['localhost'] = {
                'type': 'url_based',
                'engine': {
                    'dai_url': 'http://localhost:12345',
                    'dai_user': 'h2oai',
                    'dai_pw': 'h2oai',
                },
            }
        if await get_h2o3_localhost('http://localhost:54321'):
            q.user.user.h2o3_engines['localhost'] = {
                'type': 'url_based',
                'engine': {
                    'h2o_url': 'http://localhost:54321',
                },
            }

    q.user.last_steam_refresh = datetime.now()
    return connected


def is_valid_steam_engine_name(name: str) -> bool:
    # At least 3 characters long
    if name is None or len(name) < 3:
        return False

    # Starts with a letter
    if name[0] not in ascii_letters:
        return False

    # Allowed characters: a-z, A-Z, 0-9, _, -
    valid_chars = ascii_letters + digits + '_-'
    if not all([x in valid_chars for x in name]):
        return False

    return True


def validate_steam_engine_name(name, existing_engines):
    if not is_valid_steam_engine_name(name):
        return 'Please enter a valid name for your Steam Engine!'
    elif name == 'localhost':
        return 'Can not use "localhost" as a name for Steam Engine. Please select a different name!'
    elif name in existing_engines:
        return (
            'There is already an Engine with that name. Please select a different name!'
        )


def make_user_steam_connection(q: Q):
    if in_cloud():
        steam_credentials = SteamCredentials(access_token=q.auth.access_token)  # type: ignore
    else:
        steam_credentials = SteamCredentials(  # type: ignore
            steam_url=os.getenv('DEV_STEAM_URL', q.app.env_vars.get('DEV_STEAM_URL')),
            steam_api_token=os.getenv(
                'STEAM_API_TOKEN', q.app.env_vars.get('STEAM_API_TOKEN')
            ),
        )

    # TODO: Raise Error when this fails
    if steam_credentials.valid:
        q.user.steam_connection = create_steam_connection(
            steam_url=steam_credentials.steam_url,
            access_token=steam_credentials.access_token,
            steam_api_token=steam_credentials.steam_api_token,
        )
    else:
        q.user.steam_connection = None


def add_user(q: Q):
    user_id = q.auth.subject
    # If this user exists, do nothing
    if user_id in q.app.users:
        return

    logger.info("Initializing User")

    # Create a new user
    new_user = AppUser(
        user_id=user_id, email=q.auth.username
    )

    # Set newly created user as current user
    q.user.user = new_user

    q.app.users[user_id] = new_user

