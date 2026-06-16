from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from database.db_manager import DatabaseManager
from utils.secret_store import is_protected_secret


def test_database_manager_configures_sqlite_pragmas(tmp_path):
    db_path = tmp_path / "channel_shop.db"
    manager = DatabaseManager(db_path=str(db_path))

    with manager.engine.connect() as connection:
        journal_mode = connection.execute(text("PRAGMA journal_mode")).scalar()
        busy_timeout = connection.execute(text("PRAGMA busy_timeout")).scalar()

    assert str(journal_mode).lower() == "wal"
    assert busy_timeout == 30000


def _create_legacy_identity_tables(db_path):
    import sqlite3

    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            create table channels (
                id integer primary key autoincrement,
                channel_name varchar(50) not null,
                description varchar(255)
            );
            create table shops (
                id integer primary key autoincrement,
                channel_id integer not null,
                shop_id varchar(100) not null,
                shop_name varchar(100) not null,
                shop_logo varchar(255),
                description varchar(255)
            );
            create table accounts (
                id integer primary key autoincrement,
                shop_id integer not null,
                user_id varchar(100) not null,
                username varchar(100) not null,
                password varchar(255) not null,
                cookies text,
                status integer
            );
            create table product_knowledge (
                id integer primary key autoincrement,
                shop_id integer not null,
                goods_id integer not null,
                goods_name varchar(255) not null,
                price varchar(50),
                price_min integer,
                price_max integer,
                sold_quantity integer,
                thumb_url varchar(500),
                specifications text,
                extracted_content text,
                created_at datetime,
                updated_at datetime,
                last_extracted_at datetime
            );
            create table transfer_target_configs (
                id integer primary key autoincrement,
                shop_id integer not null,
                source_user_id varchar(100) not null,
                target_user_id varchar(100) not null,
                target_username varchar(100),
                created_at datetime,
                updated_at datetime
            );
            """
        )
        connection.commit()
    finally:
        connection.close()


def test_account_password_and_cookies_are_protected_at_rest(tmp_path):
    db_path = tmp_path / "channel_shop.db"
    manager = DatabaseManager(db_path=str(db_path))
    manager.add_shop("pinduoduo", "shop-1", "测试店铺", "logo")

    assert manager.add_account(
        "pinduoduo",
        "shop-1",
        "user-1",
        "demo-user",
        "plain-password",
        '{"PDDAccessToken":"plain-cookie"}',
    )

    with manager.engine.connect() as connection:
        row = connection.execute(
            text("select password, cookies from accounts where user_id = :user_id"),
            {"user_id": "user-1"},
        ).one()

    assert row.password != "plain-password"
    assert "plain-cookie" not in row.cookies
    assert is_protected_secret(row.password)
    assert is_protected_secret(row.cookies)

    account = manager.get_account("pinduoduo", "shop-1", "user-1")
    assert account["password"] == "plain-password"
    assert account["cookies"] == '{"PDDAccessToken":"plain-cookie"}'


def test_database_manager_accepts_current_directory_db_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    manager = DatabaseManager(db_path="channel_shop.db")

    assert (tmp_path / "channel_shop.db").exists()
    assert manager.get_channel("pinduoduo")["channel_name"] == "pinduoduo"


def test_session_scope_sanitizes_database_error_logs(tmp_path):
    class CapturingLogger:
        def __init__(self):
            self.messages = []

        def error(self, message, *args, **kwargs):
            self.messages.append(message)

    db_path = tmp_path / "channel_shop.db"
    manager = DatabaseManager(db_path=str(db_path))
    manager.logger = CapturingLogger()

    try:
        with manager.session_scope():
            raise SQLAlchemyError(
                "password=db-secret Authorization: Bearer bearer-secret"
            )
    except SQLAlchemyError:
        pass
    else:
        raise AssertionError("session_scope should re-raise database errors")

    assert manager.logger.messages
    assert "db-secret" not in manager.logger.messages[-1]
    assert "bearer-secret" not in manager.logger.messages[-1]
    assert "password=" in manager.logger.messages[-1]
    assert "Authorization" in manager.logger.messages[-1]


def test_legacy_plaintext_account_values_remain_readable(tmp_path):
    db_path = tmp_path / "channel_shop.db"
    manager = DatabaseManager(db_path=str(db_path))
    manager.add_shop("pinduoduo", "shop-1", "测试店铺", "logo")

    with manager.session_scope() as session:
        channel = manager._get_channel(session, "pinduoduo")
        shop = manager._get_shop(session, channel, "shop-1")
        from database.models import Account

        session.add(
            Account(
                shop_id=shop.id,
                user_id="legacy-user",
                username="legacy",
                password="legacy-password",
                cookies='{"legacy":"cookie"}',
            )
        )

    account = manager.get_account("pinduoduo", "shop-1", "legacy-user")
    assert account["password"] == "legacy-password"
    assert account["cookies"] == '{"legacy":"cookie"}'


def test_corrupted_protected_account_secret_does_not_crash_read_api(tmp_path):
    class CapturingLogger:
        def __init__(self):
            self.messages = []

        def info(self, message, *args, **kwargs):
            self.messages.append(str(message))

        def warning(self, message, *args, **kwargs):
            self.messages.append(str(message))

        def error(self, message, *args, **kwargs):
            self.messages.append(str(message))

    db_path = tmp_path / "channel_shop.db"
    manager = DatabaseManager(db_path=str(db_path))
    manager.logger = CapturingLogger()
    manager.add_shop("pinduoduo", "shop-1", "测试店铺", "logo")

    with manager.session_scope() as session:
        channel = manager._get_channel(session, "pinduoduo")
        shop = manager._get_shop(session, channel, "shop-1")
        from database.models import Account

        session.add(
            Account(
                shop_id=shop.id,
                user_id="broken-user",
                username="broken",
                password="dpapi:v1:not-base64",
                cookies="dpapi:v1:not-base64",
            )
        )

    account = manager.get_account("pinduoduo", "shop-1", "broken-user")

    assert account["password"] == ""
    assert account["cookies"] == ""
    joined = "\n".join(manager.logger.messages)
    assert "账号密钥读取失败" in joined
    assert "not-base64" not in joined


def test_protect_existing_account_secrets_migrates_legacy_plaintext_rows(tmp_path):
    db_path = tmp_path / "channel_shop.db"
    manager = DatabaseManager(db_path=str(db_path))
    manager.add_shop("pinduoduo", "shop-1", "测试店铺", "logo")

    with manager.session_scope() as session:
        channel = manager._get_channel(session, "pinduoduo")
        shop = manager._get_shop(session, channel, "shop-1")
        from database.models import Account

        session.add(
            Account(
                shop_id=shop.id,
                user_id="legacy-user",
                username="legacy",
                password="legacy-password",
                cookies='{"legacy":"cookie"}',
            )
        )

    assert manager.protect_existing_account_secrets() == 1
    assert manager.protect_existing_account_secrets() == 0

    with manager.engine.connect() as connection:
        row = connection.execute(
            text("select password, cookies from accounts where user_id = :user_id"),
            {"user_id": "legacy-user"},
        ).one()

    assert is_protected_secret(row.password)
    assert is_protected_secret(row.cookies)
    assert "legacy-password" not in row.password
    assert "legacy" not in row.cookies

    account = manager.get_account("pinduoduo", "shop-1", "legacy-user")
    assert account["password"] == "legacy-password"
    assert account["cookies"] == '{"legacy":"cookie"}'


def test_account_read_apis_return_external_shop_id_not_internal_row_id(tmp_path):
    db_path = tmp_path / "channel_shop.db"
    manager = DatabaseManager(db_path=str(db_path))
    external_shop_id = "pdd-shop-888"
    manager.add_shop("pinduoduo", external_shop_id, "测试店铺", "logo")
    manager.add_account(
        "pinduoduo",
        external_shop_id,
        "user-1",
        "demo-user",
        "plain-password",
        "{}",
    )

    account = manager.get_account("pinduoduo", external_shop_id, "user-1")
    accounts = manager.get_accounts_by_shop("pinduoduo", external_shop_id)

    assert account["shop_id"] == external_shop_id
    assert accounts[0]["shop_id"] == external_shop_id


def test_shop_identity_is_channel_and_external_shop_id(tmp_path):
    db_path = tmp_path / "channel_shop.db"
    manager = DatabaseManager(db_path=str(db_path))

    assert manager.add_shop("pinduoduo", "shop-1", "测试店铺", "logo")
    assert not manager.add_shop("pinduoduo", "shop-1", "重复店铺", "logo")

    shops = manager.get_shops_by_channel("pinduoduo")

    assert [shop["shop_id"] for shop in shops] == ["shop-1"]


def test_update_account_cookies_accepts_dict_and_stores_protected_json(tmp_path):
    db_path = tmp_path / "channel_shop.db"
    manager = DatabaseManager(db_path=str(db_path))
    manager.add_shop("pinduoduo", "shop-1", "测试店铺", "logo")
    manager.add_account(
        "pinduoduo",
        "shop-1",
        "user-1",
        "demo-user",
        "plain-password",
        "{}",
    )

    assert manager.update_account_cookies(
        "pinduoduo",
        "shop-1",
        "user-1",
        {"PDDAccessToken": "plain-cookie", "a": "b"},
    )

    with manager.engine.connect() as connection:
        row = connection.execute(
            text("select cookies from accounts where user_id = :user_id"),
            {"user_id": "user-1"},
        ).one()

    assert is_protected_secret(row.cookies)
    assert "plain-cookie" not in row.cookies

    account = manager.get_account("pinduoduo", "shop-1", "user-1")
    assert account["cookies"] == '{"PDDAccessToken":"plain-cookie","a":"b"}'


def test_add_account_identity_is_shop_and_user_id_not_username(tmp_path):
    db_path = tmp_path / "channel_shop.db"
    manager = DatabaseManager(db_path=str(db_path))
    manager.add_shop("pinduoduo", "shop-1", "测试店铺", "logo")

    assert manager.add_account("pinduoduo", "shop-1", "user-1", "same-name", "pw1", "{}")
    assert manager.add_account("pinduoduo", "shop-1", "user-2", "same-name", "pw2", "{}")
    assert not manager.add_account("pinduoduo", "shop-1", "user-1", "renamed", "pw3", "{}")

    accounts = manager.get_accounts_by_shop("pinduoduo", "shop-1")

    assert sorted(account["user_id"] for account in accounts) == ["user-1", "user-2"]


def test_deduplicate_shop_and_account_identities_merges_legacy_duplicates(tmp_path):
    db_path = tmp_path / "channel_shop.db"
    _create_legacy_identity_tables(db_path)

    from sqlalchemy import create_engine

    legacy_engine = create_engine(f"sqlite:///{db_path}")
    with legacy_engine.begin() as connection:
        connection.execute(
            text("insert into channels (id, channel_name, description) values (1, 'pinduoduo', '拼多多')")
        )
        connection.execute(
            text(
                """
                insert into shops (id, channel_id, shop_id, shop_name, shop_logo, description)
                values
                    (10, 1, 'shop-1', '主店', 'logo-a', null),
                    (11, 1, 'shop-1', '重复店', 'logo-b', '补充描述')
                """
            )
        )
        connection.execute(
            text(
                """
                insert into accounts (id, shop_id, user_id, username, password, cookies, status)
                values
                    (20, 10, 'user-1', 'main-user', 'pw-main', null, null),
                    (21, 10, 'user-1', 'dupe-user', 'pw-dupe', '{"from":"same-shop"}', 1),
                    (22, 11, 'user-1', 'dupe-shop-user', 'pw-shop-dupe', '{"from":"dupe-shop"}', 0),
                    (23, 11, 'user-2', 'second-user', 'pw-second', '{}', 3)
                """
            )
        )
        connection.execute(
            text(
                """
                insert into product_knowledge (id, shop_id, goods_id, goods_name, price)
                values
                    (30, 10, 1001, '主商品', '10'),
                    (31, 11, 1001, '重复商品', '12'),
                    (32, 11, 1002, '新商品', '20')
                """
            )
        )
        connection.execute(
            text(
                """
                insert into transfer_target_configs
                    (id, shop_id, source_user_id, target_user_id, target_username)
                values
                    (40, 10, 'user-1', 'target-main', '主客服'),
                    (41, 11, 'user-1', 'target-dupe', '重复客服'),
                    (42, 11, 'user-2', 'target-second', '第二客服')
                """
            )
        )

    manager = DatabaseManager(db_path=str(db_path))

    summary = manager.deduplicate_shop_and_account_identities()

    assert summary["shops_merged"] == 1
    assert summary["accounts_merged"] == 2
    assert summary["product_knowledge_merged"] == 1
    assert summary["transfer_targets_merged"] == 1
    assert summary["unique_indexes_created"] == 2

    shops = manager.get_shops_by_channel("pinduoduo")
    assert len(shops) == 1
    assert shops[0]["id"] == 10
    assert shops[0]["description"] == "补充描述"

    accounts = sorted(
        manager.get_accounts_by_shop("pinduoduo", "shop-1"),
        key=lambda account: account["user_id"],
    )
    assert [account["user_id"] for account in accounts] == ["user-1", "user-2"]
    assert accounts[0]["username"] == "main-user"
    assert accounts[0]["cookies"] == '{"from":"same-shop"}'
    assert accounts[0]["status"] == 1
    assert accounts[1]["username"] == "second-user"

    with manager.engine.connect() as connection:
        product_rows = connection.execute(
            text("select shop_id, goods_id, goods_name from product_knowledge order by goods_id")
        ).all()
        transfer_rows = connection.execute(
            text("select shop_id, source_user_id, target_user_id from transfer_target_configs order by source_user_id")
        ).all()
        duplicate_shop_count = connection.execute(
            text("select count(*) from shops where channel_id = 1 and shop_id = 'shop-1'")
        ).scalar_one()
        duplicate_account_count = connection.execute(
            text("select count(*) from accounts where shop_id = 10 and user_id = 'user-1'")
        ).scalar_one()

    assert duplicate_shop_count == 1
    assert duplicate_account_count == 1
    assert [(row.shop_id, row.goods_id, row.goods_name) for row in product_rows] == [
        (10, 1001, "主商品"),
        (10, 1002, "新商品"),
    ]
    assert [(row.shop_id, row.source_user_id, row.target_user_id) for row in transfer_rows] == [
        (10, "user-1", "target-main"),
        (10, "user-2", "target-second"),
    ]

    with manager.engine.begin() as connection:
        try:
            connection.execute(
                text(
                    """
                    insert into shops (channel_id, shop_id, shop_name, shop_logo)
                    values (1, 'shop-1', '再次重复', 'logo')
                    """
                )
            )
        except IntegrityError:
            pass
        else:
            raise AssertionError("legacy shops table should have a unique identity index after migration")

    with manager.engine.begin() as connection:
        try:
            connection.execute(
                text(
                    """
                    insert into accounts (shop_id, user_id, username, password, cookies)
                    values (10, 'user-1', 'again', 'pw', '{}')
                    """
                )
            )
        except IntegrityError:
            pass
        else:
            raise AssertionError("legacy accounts table should have a unique identity index after migration")


def test_deduplicate_shop_and_account_identities_is_idempotent(tmp_path):
    db_path = tmp_path / "channel_shop.db"
    manager = DatabaseManager(db_path=str(db_path))
    manager.add_shop("pinduoduo", "shop-1", "测试店铺", "logo")
    manager.add_account("pinduoduo", "shop-1", "user-1", "demo-user", "pw", "{}")

    assert manager.deduplicate_shop_and_account_identities() == {
        "shops_merged": 0,
        "accounts_merged": 0,
        "product_knowledge_merged": 0,
        "transfer_targets_merged": 0,
        "rows_reassigned": 0,
        "unique_indexes_created": 0,
    }
