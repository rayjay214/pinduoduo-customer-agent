import os
import json
from contextlib import contextmanager
from sqlalchemy import create_engine, event, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Any, Optional, Union, Generator
from utils.logger_loguru import get_logger
from utils.secret_store import SecretStoreError, is_protected_secret, protect_secret, reveal_secret
from utils.transfer_target import normalize_cs_uid
from core.base_service import _sanitize_for_log
from database.models import (
    Base,
    Channel,
    Shop,
    Account,
    Keyword,
    ProductKnowledge,
    CustomerServiceKnowledge,
    KnowledgeMetaEntry,
    PresaleKnowledge,
    InsaleKnowledge,
    AftersaleKnowledge,
    SceneKnowledgeEmbedding,
    TransferTargetConfig,
)


class DatabaseManager:
    """数据库管理类，提供数据库操作的封装

    单例管理：通过 DI 容器注册为单例（推荐方式）。
    也支持通过 get_db_manager() 函数获取单例实例。
    """

    def __init__(self, db_path: str = './temp/channel_shop.db'):
        """初始化数据库连接

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path

        # 确保数据库目录存在；纯文件名路径表示当前目录，不需要创建空目录。
        db_dir = os.path.dirname(os.fspath(db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        # 创建数据库引擎。SQLite 在客服系统里会被多个后台任务同时读写，
        # 这里统一设置等待锁释放的时间，并在每个连接上启用 WAL/busy_timeout。
        self.engine = create_engine(
            f'sqlite:///{db_path}',
            connect_args={"timeout": 30},
        )
        self._configure_sqlite_pragmas(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # 创建表结构
        Base.metadata.create_all(self.engine)

        self.logger = get_logger()
        # 初始化数据库
        self.init_db()

    @staticmethod
    def _configure_sqlite_pragmas(engine) -> None:
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
            finally:
                cursor.close()

    def init_db(self):
        """初始化渠道信息"""
        channel_name = "pinduoduo"
        description = "拼多多"
        self.add_channel(channel_name, description)


    def get_session(self):
        """获取数据库会话"""
        return self.Session()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """数据库会话上下文管理器，自动处理 commit/rollback/close"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"数据库操作失败: {_sanitize_for_log(e)}")
            raise
        finally:
            session.close()

    # ==================== 私有辅助方法 ====================
    def _get_channel(self, session: Session, channel_name: str) -> Optional[Channel]:
        """获取渠道对象"""
        return session.query(Channel).filter(Channel.channel_name == channel_name).first()

    def _get_shop(self, session: Session, channel: Channel, shop_id: str) -> Optional[Shop]:
        """获取店铺对象"""
        return session.query(Shop).filter(
            Shop.channel_id == channel.id,
            Shop.shop_id == shop_id
        ).first()

    def _get_account_by_user_id(self, session: Session, shop: Shop, user_id: str) -> Optional[Account]:
        """通过user_id获取账号对象"""
        return session.query(Account).filter(
            Account.shop_id == shop.id,
            Account.user_id == user_id
        ).first()

    def _get_account_by_username(self, session: Session, shop: Shop, username: str) -> Optional[Account]:
        """通过username获取账号对象"""
        return session.query(Account).filter(
            Account.shop_id == shop.id,
            Account.username == username
        ).first()

    def _reveal_account_secret(self, value: Optional[str], field_name: str, user_id: str) -> str:
        try:
            return reveal_secret(value) or ""
        except SecretStoreError as exc:
            self.logger.warning(
                f"账号密钥读取失败: user_id={user_id}, field={field_name}, error={_sanitize_for_log(exc)}"
            )
            return ""

    def _account_to_dict(self, account: Account, shop_id: Optional[str] = None, channel_name: Optional[str] = None) -> Dict[str, Any]:
        """Convert an Account row to API data, revealing protected secrets at the DB boundary."""
        data = {
            'id': account.id,
            'shop_id': shop_id if shop_id is not None else account.shop_id,
            'user_id': account.user_id,
            'username': account.username,
            'password': self._reveal_account_secret(account.password, "password", account.user_id),
            'cookies': self._reveal_account_secret(account.cookies, "cookies", account.user_id),
            'status': account.status
        }
        if channel_name is not None:
            data['channel_name'] = channel_name
        return data

    def protect_existing_account_secrets(self) -> int:
        """Protect legacy plaintext account passwords/cookies already stored in the DB.

        Returns the number of Account rows that were updated. This is intentionally
        explicit so production data can be backed up before running the migration.
        """
        updated_count = 0
        with self.session_scope() as session:
            for account in session.query(Account).all():
                updated = False
                if account.password and not is_protected_secret(account.password):
                    account.password = protect_secret(account.password)
                    updated = True
                if account.cookies and not is_protected_secret(account.cookies):
                    account.cookies = protect_secret(account.cookies)
                    updated = True
                if updated:
                    updated_count += 1
        return updated_count

    def deduplicate_shop_and_account_identities(self) -> Dict[str, int]:
        """Merge legacy duplicate shop/account identities after taking a DB backup.

        This is intentionally not called during normal startup. Older SQLite files may
        have been created before the current unique identity constraints existed, so an
        operator should run this explicit repair after backing up production data.
        """
        summary = {
            'shops_merged': 0,
            'accounts_merged': 0,
            'product_knowledge_merged': 0,
            'transfer_targets_merged': 0,
            'rows_reassigned': 0,
            'unique_indexes_created': 0,
        }

        with self.session_scope() as session:
            duplicate_shop_keys = (
                session.query(Shop.channel_id, Shop.shop_id)
                .group_by(Shop.channel_id, Shop.shop_id)
                .having(func.count(Shop.id) > 1)
                .all()
            )

            for channel_id, external_shop_id in duplicate_shop_keys:
                shops = (
                    session.query(Shop)
                    .filter(Shop.channel_id == channel_id, Shop.shop_id == external_shop_id)
                    .order_by(Shop.id)
                    .all()
                )
                if len(shops) < 2:
                    continue
                canonical = shops[0]
                self._deduplicate_accounts_for_shop(session, canonical.id, summary)
                for duplicate in shops[1:]:
                    self._deduplicate_accounts_for_shop(session, duplicate.id, summary)
                    self._merge_duplicate_shop(session, canonical, duplicate, summary)

            for (shop_id,) in session.query(Shop.id).all():
                self._deduplicate_accounts_for_shop(session, shop_id, summary)

            session.flush()
            summary['unique_indexes_created'] = self._ensure_identity_unique_indexes(session)

        return summary

    def _ensure_identity_unique_indexes(self, session: Session) -> int:
        created = 0
        connection = session.connection()
        index_specs = (
            ('shops', ('channel_id', 'shop_id'), 'ux_shops_channel_id_shop_id'),
            ('accounts', ('shop_id', 'user_id'), 'ux_accounts_shop_id_user_id'),
        )
        for table_name, columns, index_name in index_specs:
            if self._has_unique_index_for_columns(connection, table_name, columns):
                continue
            quoted_index = self._quote_sqlite_identifier(index_name)
            quoted_table = self._quote_sqlite_identifier(table_name)
            quoted_columns = ', '.join(self._quote_sqlite_identifier(column) for column in columns)
            connection.exec_driver_sql(
                f'CREATE UNIQUE INDEX {quoted_index} ON {quoted_table} ({quoted_columns})'
            )
            created += 1
        return created

    @staticmethod
    def _quote_sqlite_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def _has_unique_index_for_columns(self, connection: Any, table_name: str, columns: tuple[str, ...]) -> bool:
        quoted_table = self._quote_sqlite_identifier(table_name)
        for index_row in connection.exec_driver_sql(f'PRAGMA index_list({quoted_table})'):
            index_info = index_row._mapping
            if not index_info.get('unique'):
                continue
            quoted_index = self._quote_sqlite_identifier(index_info['name'])
            indexed_columns = tuple(
                column_row._mapping['name']
                for column_row in connection.exec_driver_sql(f'PRAGMA index_info({quoted_index})')
            )
            if indexed_columns == columns:
                return True
        return False

    @staticmethod
    def _is_missing_value(value: Any) -> bool:
        return value is None or value == ''

    def _fill_missing_fields(self, target: Any, source: Any, fields: List[str]) -> None:
        for field in fields:
            if self._is_missing_value(getattr(target, field, None)):
                source_value = getattr(source, field, None)
                if not self._is_missing_value(source_value):
                    setattr(target, field, source_value)

    def _merge_duplicate_shop(
        self,
        session: Session,
        canonical: Shop,
        duplicate: Shop,
        summary: Dict[str, int],
    ) -> None:
        self._fill_missing_fields(canonical, duplicate, ['shop_name', 'shop_logo', 'description'])
        self._merge_accounts_between_shops(session, canonical.id, duplicate.id, summary)
        self._merge_product_knowledge_between_shops(session, canonical.id, duplicate.id, summary)
        self._merge_transfer_targets_between_shops(session, canonical.id, duplicate.id, summary)

        for model in (
            CustomerServiceKnowledge,
            KnowledgeMetaEntry,
            PresaleKnowledge,
            InsaleKnowledge,
            AftersaleKnowledge,
            SceneKnowledgeEmbedding,
        ):
            moved = (
                session.query(model)
                .filter(model.shop_id == duplicate.id)
                .update({model.shop_id: canonical.id}, synchronize_session=False)
            )
            summary['rows_reassigned'] += moved

        session.delete(duplicate)
        summary['shops_merged'] += 1

    def _merge_accounts_between_shops(
        self,
        session: Session,
        canonical_shop_id: int,
        duplicate_shop_id: int,
        summary: Dict[str, int],
    ) -> None:
        duplicate_accounts = (
            session.query(Account)
            .filter(Account.shop_id == duplicate_shop_id)
            .order_by(Account.id)
            .all()
        )
        for account in duplicate_accounts:
            existing = (
                session.query(Account)
                .filter(Account.shop_id == canonical_shop_id, Account.user_id == account.user_id)
                .order_by(Account.id)
                .first()
            )
            if existing:
                self._merge_account_rows(existing, account)
                session.delete(account)
                summary['accounts_merged'] += 1
            else:
                account.shop_id = canonical_shop_id
                summary['rows_reassigned'] += 1

    def _deduplicate_accounts_for_shop(
        self,
        session: Session,
        shop_id: int,
        summary: Dict[str, int],
    ) -> None:
        duplicate_user_ids = (
            session.query(Account.user_id)
            .filter(Account.shop_id == shop_id)
            .group_by(Account.user_id)
            .having(func.count(Account.id) > 1)
            .all()
        )
        for (user_id,) in duplicate_user_ids:
            accounts = (
                session.query(Account)
                .filter(Account.shop_id == shop_id, Account.user_id == user_id)
                .order_by(Account.id)
                .all()
            )
            if len(accounts) < 2:
                continue
            canonical = accounts[0]
            for duplicate in accounts[1:]:
                self._merge_account_rows(canonical, duplicate)
                session.delete(duplicate)
                summary['accounts_merged'] += 1

    def _merge_account_rows(self, canonical: Account, duplicate: Account) -> None:
        self._fill_missing_fields(canonical, duplicate, ['username', 'password', 'cookies'])
        if canonical.status is None and duplicate.status is not None:
            canonical.status = duplicate.status

    def _merge_product_knowledge_between_shops(
        self,
        session: Session,
        canonical_shop_id: int,
        duplicate_shop_id: int,
        summary: Dict[str, int],
    ) -> None:
        duplicate_rows = (
            session.query(ProductKnowledge)
            .filter(ProductKnowledge.shop_id == duplicate_shop_id)
            .order_by(ProductKnowledge.id)
            .all()
        )
        for row in duplicate_rows:
            existing = (
                session.query(ProductKnowledge)
                .filter(
                    ProductKnowledge.shop_id == canonical_shop_id,
                    ProductKnowledge.goods_id == row.goods_id,
                )
                .order_by(ProductKnowledge.id)
                .first()
            )
            if existing:
                self._fill_missing_fields(
                    existing,
                    row,
                    [
                        'goods_name',
                        'price',
                        'price_min',
                        'price_max',
                        'sold_quantity',
                        'thumb_url',
                        'specifications',
                        'extracted_content',
                    ],
                )
                session.delete(row)
                summary['product_knowledge_merged'] += 1
            else:
                row.shop_id = canonical_shop_id
                summary['rows_reassigned'] += 1

    def _merge_transfer_targets_between_shops(
        self,
        session: Session,
        canonical_shop_id: int,
        duplicate_shop_id: int,
        summary: Dict[str, int],
    ) -> None:
        duplicate_rows = (
            session.query(TransferTargetConfig)
            .filter(TransferTargetConfig.shop_id == duplicate_shop_id)
            .order_by(TransferTargetConfig.id)
            .all()
        )
        for row in duplicate_rows:
            existing = (
                session.query(TransferTargetConfig)
                .filter(
                    TransferTargetConfig.shop_id == canonical_shop_id,
                    TransferTargetConfig.source_user_id == row.source_user_id,
                )
                .order_by(TransferTargetConfig.id)
                .first()
            )
            if existing:
                self._fill_missing_fields(existing, row, ['target_user_id', 'target_username'])
                session.delete(row)
                summary['transfer_targets_merged'] += 1
            else:
                row.shop_id = canonical_shop_id
                summary['rows_reassigned'] += 1

    # ==================== 渠道相关操作 ====================
    def add_channel(self, channel_name: str, description: str = None) -> bool:
        """添加渠道"""
        with self.session_scope() as session:
            existing = session.query(Channel).filter(Channel.channel_name == channel_name).first()
            if existing:
                return True
            channel = Channel(channel_name=channel_name, description=description)
            session.add(channel)
            return True

    def get_channel(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """获取渠道信息"""
        with self.session_scope() as session:
            channel = session.query(Channel).filter(Channel.channel_name == channel_name).first()
            if not channel:
                return None
            return {
                'id': channel.id,
                'channel_name': channel.channel_name,
                'description': channel.description
            }

    def get_all_channels(self) -> List[Dict[str, Any]]:
        """获取所有渠道"""
        with self.session_scope() as session:
            channels = session.query(Channel).all()
            return [
                {
                    'id': channel.id,
                    'channel_name': channel.channel_name,
                    'description': channel.description
                }
                for channel in channels
            ]

    def delete_channel(self, channel_name: str) -> bool:
        """删除渠道"""
        with self.session_scope() as session:
            channel = session.query(Channel).filter(Channel.channel_name == channel_name).first()
            if not channel:
                self.logger.warning(f"渠道 {channel_name} 不存在")
                return False
            session.delete(channel)
            self.logger.info(f"成功删除渠道: {channel_name}")
            return True
    
    # 店铺相关操作
    def add_shop(self, channel_name: str, shop_id: str, shop_name: str, shop_logo: str, description: str = None) -> bool:
        """添加店铺"""
        with self.session_scope() as session:
            channel = session.query(Channel).filter(Channel.channel_name == channel_name).first()
            if not channel:
                self.logger.error(f"添加店铺失败: 渠道 {channel_name} 不存在")
                return False
            existing = session.query(Shop).filter(
                Shop.channel_id == channel.id,
                Shop.shop_id == shop_id
            ).first()
            if existing:
                self.logger.warning(f"店铺 {shop_id} 已存在于渠道 {channel_name}")
                return False
            shop = Shop(
                channel_id=channel.id,
                shop_id=shop_id,
                shop_name=shop_name,
                shop_logo=shop_logo,
                description=description
            )
            session.add(shop)
            self.logger.info(f"成功添加店铺: {shop_name}({shop_id}) 到渠道 {channel_name}")
            return True

    def get_shop(self, channel_name: str, shop_id: str) -> Optional[Dict[str, Any]]:
        """获取店铺信息"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return None
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return None
            return {
                'id': shop.id,
                'channel_id': shop.channel_id,
                'channel_name': channel_name,
                'shop_id': shop.shop_id,
                'shop_name': shop.shop_name,
                'shop_logo': shop.shop_logo,
                'description': shop.description,
            }

    def get_shops_by_channel(self, channel_name: str) -> List[Dict[str, Any]]:
        """获取指定渠道下的所有店铺"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return []
            shops = session.query(Shop).filter(Shop.channel_id == channel.id).all()
            return [
                {
                    'id': shop.id,
                    'channel_id': shop.channel_id,
                    'channel_name': channel_name,
                    'shop_id': shop.shop_id,
                    'shop_name': shop.shop_name,
                    'shop_logo': shop.shop_logo,
                    'description': shop.description
                }
                for shop in shops
            ]

    def update_shop_info(self, channel_name: str, shop_id: str, shop_name: str = None, shop_logo: str = None, description: str = None) -> bool:
        """更新店铺信息"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return False
            if shop_name is not None:
                shop.shop_name = shop_name
            if shop_logo is not None:
                shop.shop_logo = shop_logo
            if description is not None:
                shop.description = description
            return True

    def delete_shop(self, channel_name: str, shop_id: str) -> bool:
        """删除店铺"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return False
            session.delete(shop)
            return True

    # ==================== 账号相关操作 ====================
    def add_account(self, channel_name: str, shop_id: str, user_id: str, username: str, password: str, cookies: str = None) -> bool:
        """添加账号"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                self.logger.error(f"添加账号失败: 渠道 {channel_name} 不存在")
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                self.logger.error(f"添加账号失败: 店铺 {shop_id} 不存在")
                return False
            existing = self._get_account_by_user_id(session, shop, user_id)
            if existing:
                self.logger.warning(f"账号 {user_id} 已存在于店铺 {shop_id}")
                return False
            account = Account(
                shop_id=shop.id,
                user_id=user_id,
                username=username,
                password=protect_secret(password),
                cookies=protect_secret(cookies),
                status=None
            )
            session.add(account)
            self.logger.info(f"成功添加账号: {username} 到店铺 {shop_id}")
            return True

    def get_account(self, channel_name: str, shop_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """获取账号信息"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                self.logger.warning(f"未找到渠道: {channel_name}")
                return None
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                self.logger.warning(f"未找到店铺: {shop_id} (渠道: {channel_name})")
                return None
            account = self._get_account_by_user_id(session, shop, user_id)
            if not account:
                self.logger.warning(f"未找到账户: {user_id} (店铺 ID: {shop_id})")
                return None
            return self._account_to_dict(account, shop_id=shop.shop_id)

    def update_account_info(self, channel_name: str, shop_id: str, user_id: str, username: Optional[str] = None, password: Optional[str] = None, cookies: Optional[str] = None, status: Optional[int] = None) -> bool:
        """更新账号信息"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                self.logger.error(f"更新账号失败: 渠道 {channel_name} 不存在")
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                self.logger.error(f"更新账号失败: 店铺 {shop_id} 不存在于渠道 {channel_name}")
                return False
            account = self._get_account_by_user_id(session, shop, user_id)
            if not account:
                self.logger.error(f"更新账号失败: 账号 {user_id} 不存在于店铺 {shop_id}")
                return False
            if username is not None:
                account.username = username
            if password is not None:
                account.password = protect_secret(password)
            if cookies is not None:
                account.cookies = protect_secret(cookies)
            if status is not None:
                account.status = status
            self.logger.info(f"成功更新账号信息: {username} (用户ID: {user_id})")
            return True

    def get_accounts_by_shop(self, channel_name: str, shop_id: str) -> List[Dict[str, Any]]:
        """获取指定店铺下的所有账号"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return []
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return []
            accounts = session.query(Account).filter(Account.shop_id == shop.id).all()
            return [self._account_to_dict(account, shop_id=shop.shop_id) for account in accounts]

    def get_all_accounts_with_details(self) -> List[Dict[str, Any]]:
        """
        批量获取所有账号及其关联的店铺和渠道信息（减少N+1查询）

        Returns:
            List[Dict]: 包含 channel_name, shop_id, shop_name, shop_logo, username, password, status, user_id, cookies
        """
        with self.session_scope() as session:
            # 使用 join 一次性查询所有数据
            results = (
                session.query(Account, Shop, Channel)
                .join(Shop, Account.shop_id == Shop.id)
                .join(Channel, Shop.channel_id == Channel.id)
                .all()
            )

            return [
                {
                    'channel_name': channel.channel_name,
                    'shop_id': shop.shop_id,
                    'shop_name': shop.shop_name,
                    'shop_logo': shop.shop_logo,
                    'username': account.username,
                    'password': self._reveal_account_secret(account.password, "password", account.user_id),
                    'status': account.status,
                    'user_id': account.user_id,
                    'cookies': self._reveal_account_secret(account.cookies, "cookies", account.user_id)
                }
                for account, shop, channel in results
            ]

    def update_account_status(self, channel_name: str, shop_id: str, user_id: str, status: int) -> bool:
        """更新账号状态"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return False
            account = self._get_account_by_user_id(session, shop, user_id)
            if not account:
                return False
            account.status = status
            return True

    def update_account_cookies(self, channel_name: str, shop_id: str, user_id: str, cookies: str) -> bool:
        """更新账号cookies"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return False
            account = self._get_account_by_user_id(session, shop, user_id)
            if not account:
                return False
            account.cookies = protect_secret(cookies)
            return True

    def delete_account(self, channel_name: str, shop_id: str, user_id: str) -> bool:
        """删除账号"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return False
            account = self._get_account_by_user_id(session, shop, user_id)
            if not account:
                return False
            session.delete(account)
            return True

    # ==================== 转人工目标配置 ====================
    def get_transfer_target(self, channel_name: str, shop_id: str, source_user_id: str) -> Optional[Dict[str, Any]]:
        """获取账号的优先转人工目标配置。"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return None
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return None

            config = session.query(TransferTargetConfig).filter(
                TransferTargetConfig.shop_id == shop.id,
                TransferTargetConfig.source_user_id == str(source_user_id)
            ).first()
            if not config:
                return None

            return {
                'id': config.id,
                'shop_id': shop_id,
                'source_user_id': config.source_user_id,
                'target_user_id': config.target_user_id,
                'target_username': config.target_username,
            }

    def set_transfer_target(
        self,
        channel_name: str,
        shop_id: str,
        source_user_id: str,
        target_user_id: Optional[str],
        target_username: Optional[str] = None,
    ) -> bool:
        """设置账号的优先转人工目标；target_user_id 为空时清除配置。"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                self.logger.error(f"设置转人工目标失败: 渠道 {channel_name} 不存在")
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                self.logger.error(f"设置转人工目标失败: 店铺 {shop_id} 不存在")
                return False

            existing = session.query(TransferTargetConfig).filter(
                TransferTargetConfig.shop_id == shop.id,
                TransferTargetConfig.source_user_id == str(source_user_id)
            ).first()

            normalized_target_user_id = normalize_cs_uid(str(shop_id), target_user_id) or ""
            normalized_target_username = target_username.strip() if isinstance(target_username, str) else target_username

            if not normalized_target_user_id:
                if existing:
                    session.delete(existing)
                    self.logger.info(
                        f"已清除转人工目标配置: 店铺={shop_id}, source_user_id={source_user_id}"
                    )
                return True

            if existing:
                existing.target_user_id = normalized_target_user_id
                existing.target_username = normalized_target_username
            else:
                existing = TransferTargetConfig(
                    shop_id=shop.id,
                    source_user_id=str(source_user_id),
                    target_user_id=normalized_target_user_id,
                    target_username=normalized_target_username,
                )
                session.add(existing)

            self.logger.info(
                f"已设置转人工目标: 店铺={shop_id}, source_user_id={source_user_id}, "
                f"target_user_id={normalized_target_user_id}, target_username={normalized_target_username}"
            )
            return True

    # 关键词相关操作
    def add_keyword(self, keyword: str) -> bool:
        """添加关键词"""
        with self.session_scope() as session:
            existing = session.query(Keyword).filter(Keyword.keyword == keyword).first()
            if existing:
                self.logger.warning(f"关键词 {keyword} 已存在")
                return False
            keyword_obj = Keyword(keyword=keyword)
            session.add(keyword_obj)
            self.logger.info(f"成功添加关键词: {keyword}")
            return True

    def get_keyword(self, keyword: str) -> Optional[Dict[str, Any]]:
        """获取关键词信息"""
        with self.session_scope() as session:
            keyword_obj = session.query(Keyword).filter(Keyword.keyword == keyword).first()
            if not keyword_obj:
                return None
            return {
                'id': keyword_obj.id,
                'keyword': keyword_obj.keyword
            }

    def get_all_keywords(self) -> List[Dict[str, Any]]:
        """获取所有关键词"""
        with self.session_scope() as session:
            keywords = session.query(Keyword).all()
            return [
                {
                    'id': keyword.id,
                    'keyword': keyword.keyword
                }
                for keyword in keywords
            ]

    def update_keyword(self, old_keyword: str, new_keyword: str) -> bool:
        """更新关键词"""
        with self.session_scope() as session:
            keyword_obj = session.query(Keyword).filter(Keyword.keyword == old_keyword).first()
            if not keyword_obj:
                self.logger.warning(f"关键词 {old_keyword} 不存在")
                return False
            if old_keyword != new_keyword:
                existing = session.query(Keyword).filter(Keyword.keyword == new_keyword).first()
                if existing:
                    self.logger.warning(f"关键词 {new_keyword} 已存在")
                    return False
            keyword_obj.keyword = new_keyword
            self.logger.info(f"成功更新关键词: {old_keyword} -> {new_keyword}")
            return True

    def delete_keyword(self, keyword: str) -> bool:
        """删除关键词"""
        with self.session_scope() as session:
            keyword_obj = session.query(Keyword).filter(Keyword.keyword == keyword).first()
            if not keyword_obj:
                self.logger.warning(f"关键词 {keyword} 不存在")
                return False
            session.delete(keyword_obj)
            self.logger.info(f"成功删除关键词: {keyword}")
            return True

_db_instance: Optional["DatabaseManager"] = None

def get_db_manager() -> "DatabaseManager":
    """获取 DatabaseManager 单例。

    优先从 DI 容器获取；若 DI 尚未注册（启动早期），回退到本地单例，
    以确保全应用始终共用同一个实例，避免出现多个 db 文件分裂。
    """
    global _db_instance
    try:
        from core.di_container import container
        if container.is_registered(DatabaseManager):
            return container.get(DatabaseManager)
    except ImportError:
        pass

    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance

class _LazyDBProxy:
    """延迟代理，用于兼容旧代码的全局 db_manager 实例，底层共用 DI 容器。"""

    def __getattr__(self, name: str):
        return getattr(get_db_manager(), name)

db_manager = _LazyDBProxy()
