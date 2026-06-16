from database import knowledge_service as knowledge_service_module
from database import knowledge_service as knowledge_service_module
from database.knowledge_service import KnowledgeService
from database.models import Base, PresaleKnowledge
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class FakeSession:
    def __init__(self):
        self.added = []
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def scalar(self, _stmt):
        return None

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True


def test_batch_import_customer_service_skips_malformed_rows():
    session = FakeSession()
    service = object.__new__(KnowledgeService)
    service.get_session = lambda: session

    success, skipped = service.batch_import_customer_service(
        1,
        [
            "bad-row",
            {"title": "", "content": "无标题"},
            {"title": "无内容", "content": ""},
            {"title": "有效标题", "content": "有效内容", "tags": "tag"},
        ],
    )

    assert success == 1
    assert skipped == 3
    assert session.committed is True
    assert len(session.added) == 1
    assert session.added[0].title == "有效标题"


def test_get_all_tags_tolerates_malformed_rows():
    class TagSession(FakeSession):
        def execute(self, _stmt):
            return [
                ("售前, 参数",),
                (123,),
                (None,),
                (),
                "bad-row",
            ]

    service = object.__new__(KnowledgeService)
    service.get_session = lambda: TagSession()

    assert service.get_all_tags(1) == ["123", "参数", "售前"]


def test_add_customer_service_parses_string_false_enabled():
    session = FakeSession()
    service = object.__new__(KnowledgeService)
    service.get_session = lambda: session

    service.add_customer_service(
        shop_id=1,
        title="标题",
        content="内容",
        enabled="false",
    )

    assert session.added[0].enabled is False


def test_update_customer_service_parses_string_false_enabled():
    row = type("Row", (), {"enabled": True, "title": "旧标题", "content": "旧内容", "tags": None})()

    class UpdateSession(FakeSession):
        def get(self, _model, _key):
            return row

    service = object.__new__(KnowledgeService)
    service.get_session = lambda: UpdateSession()

    service.update_customer_service(1, enabled="false")

    assert row.enabled is False


def test_update_customer_service_invalid_enabled_preserves_string_false():
    row = type("Row", (), {"enabled": "false", "title": "旧标题", "content": "旧内容", "tags": None})()

    class UpdateSession(FakeSession):
        def get(self, _model, _key):
            return row

    service = object.__new__(KnowledgeService)
    service.get_session = lambda: UpdateSession()

    service.update_customer_service(1, enabled="not-a-bool")

    assert row.enabled is False


def test_build_scene_embeddings_treats_short_embedding_batch_as_failed(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add_all(
            [
                PresaleKnowledge(
                    shop_id=1,
                    aliases="问法A",
                    answer="答案A",
                    section_title="标题A",
                    enabled=True,
                ),
                PresaleKnowledge(
                    shop_id=1,
                    aliases="问法B",
                    answer="答案B",
                    section_title="标题B",
                    enabled=True,
                ),
            ]
        )
        session.commit()

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    monkeypatch.setattr("requests.post", lambda *_args, **_kwargs: FakeResponse())
    service = object.__new__(KnowledgeService)
    service.get_session = Session
    service.vector_retriever = type(
        "Retriever",
        (),
        {
            "embedding_model": "fake-model",
            "embedding_url": "http://embedding.test",
            "timeout_seconds": 1,
        },
    )()

    stats = service.build_scene_embeddings(scene="presale", batch_size=2)

    assert stats["created"] == 0
    assert stats["failed"] == 2


def test_build_scene_embeddings_masks_sensitive_batch_exception(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(
            PresaleKnowledge(
                shop_id=1,
                aliases="问法A",
                answer="答案A",
                section_title="标题A",
                enabled=True,
            )
        )
        session.commit()

    messages = []

    class FakeLogger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, message):
            messages.append(str(message))

        def debug(self, message):
            messages.append(str(message))

    def fail_post(*_args, **_kwargs):
        raise RuntimeError("token=secret-token")

    monkeypatch.setattr("requests.post", fail_post)
    monkeypatch.setattr(knowledge_service_module, "logger", FakeLogger())

    service = object.__new__(KnowledgeService)
    service.get_session = Session
    service.vector_retriever = type(
        "Retriever",
        (),
        {
            "embedding_model": "fake-model",
            "embedding_url": "http://embedding.test",
            "timeout_seconds": 1,
        },
    )()

    stats = service.build_scene_embeddings(scene="presale", batch_size=1)

    joined = "\n".join(messages)
    assert stats["failed"] == 1
    assert "embedding批次失败" in joined
    assert "secret-token" not in joined
    assert "token=***" in joined


def test_build_scene_embeddings_masks_sensitive_batch_exception(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(
            PresaleKnowledge(
                shop_id=1,
                aliases="问法A",
                answer="答案A",
                section_title="标题A",
                enabled=True,
            )
        )
        session.commit()

    class FakeLogger:
        def __init__(self):
            self.messages = []

        def info(self, message):
            self.messages.append(str(message))

        def warning(self, message):
            self.messages.append(str(message))

        def debug(self, message):
            self.messages.append(str(message))

    fake_logger = FakeLogger()

    def broken_post(*_args, **_kwargs):
        raise RuntimeError("token=secret-token")

    monkeypatch.setattr("requests.post", broken_post)
    monkeypatch.setattr(knowledge_service_module, "logger", fake_logger)
    service = object.__new__(KnowledgeService)
    service.get_session = Session
    service.vector_retriever = type(
        "Retriever",
        (),
        {
            "embedding_model": "fake-model",
            "embedding_url": "http://embedding.test",
            "timeout_seconds": 1,
        },
    )()

    stats = service.build_scene_embeddings(scene="presale", batch_size=1)

    joined = "\n".join(fake_logger.messages)
    assert stats["created"] == 0
    assert stats["failed"] == 1
    assert "[embedding批次失败]" in joined
    assert "secret-token" not in joined
    assert "token=***" in joined
