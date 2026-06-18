"""
知识库管理UI模块
==============

提供产品知识和客服知识管理界面，包含：
- 顶部店铺选择器
- 两个标签页：产品知识 / 客服知识
- 自动同步产品知识（拼多多API + LLM提取）
- 客服知识人工添加/编辑/删除
"""
from __future__ import annotations
import asyncio
import os
from typing import TYPE_CHECKING, Optional, List, Dict
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QStackedWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QPushButton, QMessageBox, QDialog, QDialogButtonBox,
    QLineEdit, QTextEdit, QCheckBox, QProgressBar, QFrame, QFileDialog, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from qfluentwidgets import (
    PrimaryPushButton, PushButton,
    InfoBar, InfoBarPosition, TableWidget, SegmentedWidget,
    ComboBox,
)

from core.di_container import container
from database.knowledge_service import KnowledgeService
from database.product_sync import ProductSyncService, SyncProgress
from database.models import ProductKnowledge, CustomerServiceKnowledge, KnowledgeMetaEntry, Shop
from utils.config_values import as_bool, as_int
from utils.logger_loguru import get_logger

if TYPE_CHECKING:
    from database.knowledge_service import KnowledgeService

logger = get_logger("KnowledgeUI")


SCENE_KNOWLEDGE_LIGHT_STYLE = """
QDialog, QWidget {
    background: #ffffff;
    color: #111111;
}
QLabel, QCheckBox {
    color: #111111;
}
QLineEdit, QTextEdit, QSpinBox {
    background: #ffffff;
    color: #111111;
    border: 1px solid #c9cdd4;
    border-radius: 4px;
    selection-background-color: #2b73d2;
    selection-color: #ffffff;
}
QLineEdit:disabled, QTextEdit:disabled, QSpinBox:disabled {
    background: #f3f4f6;
    color: #666666;
}
QTableWidget, QTableView {
    background: #ffffff;
    alternate-background-color: #f7f8fa;
    color: #111111;
    gridline-color: #d8dce3;
    selection-background-color: #dcecff;
    selection-color: #111111;
}
QHeaderView::section {
    background: #f0f2f5;
    color: #111111;
    border: 1px solid #d8dce3;
    padding: 6px;
}
QTableCornerButton::section {
    background: #f0f2f5;
    border: 1px solid #d8dce3;
}
"""


class SyncWorker(QThread):
    """同步工作线程"""
    progress_updated = pyqtSignal(int, int, int, str, str)  # current, total, success, current_name, phase
    sync_finished = pyqtSignal(int, int, bool)  # success, failed, cancelled

    def __init__(
        self,
        shop_db_id: int,
        pdd_shop_id: str,
        user_id: str,
        is_full_sync: bool,
        product_sync: ProductSyncService,
        parent=None,
    ):
        super().__init__(parent)
        self.shop_db_id = shop_db_id
        self.pdd_shop_id = pdd_shop_id
        self.user_id = user_id
        self.is_full_sync = is_full_sync
        self.product_sync = product_sync
        self._cancelled = False

    def run(self):
        """运行同步"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def progress_callback(progress: SyncProgress):
            self.progress_updated.emit(
                progress.current,
                progress.total,
                progress.success,
                progress.current_goods_name,
                progress.phase,
            )

        try:
            result = loop.run_until_complete(
                self.product_sync.sync_shop(
                    shop_id=self.pdd_shop_id,
                    shop_db_id=self.shop_db_id,
                    user_id=self.user_id,
                    is_full_sync=self.is_full_sync,
                    progress_callback=progress_callback,
                )
            )
            self.sync_finished.emit(result.success, result.failed, result.cancelled)
        except Exception as exc:
            logger.error(f"产品知识同步线程异常: {exc}")
            self.sync_finished.emit(0, 1, False)
        finally:
            loop.close()


class IndexRebuildWorker(QThread):
    """知识库索引重构线程。"""

    rebuild_finished = pyqtSignal(bool, str)

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        shop_id: Optional[int],
        parent=None,
    ):
        super().__init__(parent)
        self.knowledge_service = knowledge_service
        self.shop_id = shop_id

    def run(self):
        try:
            stats = self.knowledge_service.rebuild_knowledge_index(
                shop_id=self.shop_id,
                batch_size=128,
            )
            deleted = stats.get("_deleted_embeddings", 0)
            parts = [f"已清理旧索引 {deleted} 条"]
            for scene_key, label in (("presale", "售前"), ("insale", "售中"), ("aftersale", "售后")):
                item = stats.get(scene_key, {}) or {}
                parts.append(
                    "{}: 复用{} 新建{} 失败{}".format(
                        label,
                        item.get("skipped", 0),
                        item.get("created", 0),
                        item.get("failed", 0),
                    )
                )
            self.rebuild_finished.emit(True, "；".join(parts))
        except Exception as exc:
            logger.error(f"知识库索引重构失败: {exc}")
            self.rebuild_finished.emit(False, str(exc))


class ProductDetailDialog(QDialog):
    """产品知识详情对话框，支持编辑"""

    def __init__(self, product: ProductKnowledge, parent=None):
        super().__init__(parent)
        self.product = product
        self.setWindowTitle("产品知识详情")
        self.resize(700, 500)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 商品名称
        self.name_label = QLabel("商品名称:")
        self.name_edit = QLineEdit(self.product.goods_name)
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_edit)

        # 提取内容
        self.content_label = QLabel("提取知识:")
        self.content_edit = QTextEdit()
        self.content_edit.setPlainText(self.product.extracted_content or "")
        self.content_edit.setPlaceholderText("LLM提取的产品知识会显示在这里，你可以手动编辑")
        layout.addWidget(self.content_label)
        layout.addWidget(self.content_edit)

        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        """获取编辑后的数据"""
        return {
            "goods_name": self.name_edit.text().strip(),
            "extracted_content": self.content_edit.toPlainText().strip(),
        }


class CsAddEditDialog(QDialog):
    """客服知识添加/编辑对话框"""

    def __init__(
        self,
        shop_id: int,
        existing: Optional[CustomerServiceKnowledge] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.shop_id = shop_id
        self.existing = existing
        self.default_tags = ["物流", "售后", "支付", "商品规格", "优惠券", "会员", "发货时间", "退换货"]
        self.setWindowTitle("添加客服知识" if not existing else "编辑客服知识")
        self.resize(650, 500)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        self.title_label = QLabel("标题:")
        self.title_edit = QLineEdit()
        if self.existing:
            self.title_edit.setText(self.existing.title)
        layout.addWidget(self.title_label)
        layout.addWidget(self.title_edit)

        # 内容
        self.content_label = QLabel("内容:")
        self.content_edit = QTextEdit()
        self.content_edit.setPlaceholderText("输入客服知识内容，例如：退换货政策说明...")
        if self.existing:
            self.content_edit.setText(self.existing.content)
        layout.addWidget(self.content_label)
        layout.addWidget(self.content_edit)

        # 标签 - 预设复选框
        self.tags_label = QLabel("选择标签 (可多选):")
        layout.addWidget(self.tags_label)

        self.tag_checkboxes: List[QCheckBox] = []
        existing_tags = []
        if self.existing and self.existing.tags:
            existing_tags = [t.strip() for t in self.existing.tags.split(',') if t.strip()]

        tags_frame = QFrame()
        tags_layout = QHBoxLayout(tags_frame)
        tags_layout.setSpacing(8)

        for tag in self.default_tags:
            cb = QCheckBox(tag)
            if tag in existing_tags:
                cb.setChecked(True)
            tags_layout.addWidget(cb)
            self.tag_checkboxes.append(cb)

        layout.addWidget(tags_frame)

        # 自定义标签
        self.custom_label = QLabel("自定义标签 (逗号分隔):")
        self.custom_edit = QLineEdit()
        if self.existing and self.existing.tags:
            # 已有标签中不在预设列表的合并到自定义
            existing_custom = [
                t for t in existing_tags
                if t not in self.default_tags
            ]
            if existing_custom:
                self.custom_edit.setText(','.join(existing_custom))
        layout.addWidget(self.custom_label)
        layout.addWidget(self.custom_edit)

        # 启用
        self.enabled_cb = QCheckBox("启用")
        self.enabled_cb.setChecked(True)
        if self.existing:
            self.enabled_cb.setChecked(self.existing.enabled)
        layout.addWidget(self.enabled_cb)

        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        """获取数据"""
        title = self.title_edit.text().strip()
        content = self.content_edit.toPlainText().strip()
        enabled = self.enabled_cb.isChecked()

        # 收集选中的预设标签
        selected_tags = [
            cb.text() for cb in self.tag_checkboxes
                if cb.isChecked()
        ]

        # 添加自定义标签
        custom = self.custom_edit.text().strip()
        if custom:
            selected_tags.extend([t.strip() for t in custom.split(',') if t.strip()])

        # 去重
        selected_tags = list(dict.fromkeys(selected_tags))
        tags_str = ','.join(selected_tags) if selected_tags else None

        return {
            "title": title,
            "content": content,
            "tags": tags_str,
            "enabled": enabled,
        }


class SceneKnowledgeEditDialog(QDialog):
    """场景知识编辑对话框。"""

    def __init__(self, entry: Dict, parent=None):
        super().__init__(parent)
        self.entry = entry
        self.setWindowTitle("编辑场景知识")
        self.resize(760, 560)
        self.setStyleSheet(SCENE_KNOWLEDGE_LIGHT_STYLE)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        self.section_edit = QLineEdit(self.entry.get("section_title", ""))
        self.intent_edit = QLineEdit(self.entry.get("sub_intent", ""))
        self.aliases_edit = QTextEdit()
        self.aliases_edit.setPlainText(self.entry.get("aliases", ""))
        self.aliases_edit.setPlaceholderText("多个问法用 / 分隔，例如：续航多久/能用多久/充满电能用多久")
        self.answer_edit = QTextEdit()
        self.answer_edit.setPlainText(self.entry.get("answer", ""))
        self.answer_edit.setPlaceholderText("填写客服可直接参考的标准答案")

        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(0, 999)
        self.priority_spin.setValue(as_int(self.entry.get("priority", 0), 0))
        self.enabled_cb = QCheckBox("启用")
        self.enabled_cb.setChecked(as_bool(self.entry.get("enabled", True), True))

        layout.addWidget(QLabel("分类标题:"))
        layout.addWidget(self.section_edit)
        layout.addWidget(QLabel("细分意图:"))
        layout.addWidget(self.intent_edit)
        layout.addWidget(QLabel("问法 aliases（用 / 分隔）:"))
        layout.addWidget(self.aliases_edit, 1)
        layout.addWidget(QLabel("答案 answer:"))
        layout.addWidget(self.answer_edit, 2)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("优先级:"))
        bottom.addWidget(self.priority_spin)
        bottom.addWidget(self.enabled_cb)
        bottom.addStretch()
        layout.addLayout(bottom)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self) -> Dict:
        return {
            "section_title": self.section_edit.text().strip(),
            "sub_intent": self.intent_edit.text().strip(),
            "aliases": self.aliases_edit.toPlainText().strip(),
            "answer": self.answer_edit.toPlainText().strip(),
            "priority": self.priority_spin.value(),
            "enabled": self.enabled_cb.isChecked(),
        }


class SceneKnowledgeListDialog(QDialog):
    """某个商品/店铺通用场景的知识列表。"""

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        shop_id: int,
        goods_id: Optional[int],
        goods_name: str,
        scene_key: str,
        parent=None,
    ):
        super().__init__(parent)
        self.knowledge_service = knowledge_service
        self.shop_id = shop_id
        self.goods_id = goods_id
        self.goods_name = goods_name
        self.scene_key = scene_key
        self.entries: List[Dict] = []
        scene_label = {"presale": "售前", "insale": "售中", "aftersale": "售后"}.get(scene_key, scene_key)
        if goods_id is None:
            self.setWindowTitle(f"{scene_label}知识 - 店铺通用")
        else:
            self.setWindowTitle(f"{scene_label}知识 - {goods_id}")
        self.resize(1100, 720)
        self.setStyleSheet(SCENE_KNOWLEDGE_LIGHT_STYLE)
        self._init_ui()
        self._refresh_entries()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()
        if self.goods_id is None:
            header_text = "店铺通用知识（对所有商品生效）"
        else:
            header_text = f"{self.goods_id}  {self.goods_name}"
        title = QLabel(header_text)
        header_layout.addWidget(title)
        header_layout.addStretch()

        add_btn = PrimaryPushButton("添加")
        add_btn.clicked.connect(self._add_entry)
        header_layout.addWidget(add_btn)
        layout.addLayout(header_layout)

        self.table = TableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["分类", "细分意图", "优先级", "问法", "答案", "编辑", "删除"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(5, 70)
        self.table.setColumnWidth(6, 70)
        self.table.cellDoubleClicked.connect(lambda row, _: self._edit_entry(row))
        layout.addWidget(self.table, 1)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText("点击知识行后，这里显示完整问法和答案")
        self.table.cellClicked.connect(self._show_entry_detail)
        layout.addWidget(self.detail)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_entry(self):
        dialog = SceneKnowledgeEditDialog({}, self.window())
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        data = dialog.get_data()
        if not data["aliases"] or not data["answer"]:
            QMessageBox.warning(self, "保存失败", "问法和答案不能为空")
            return
        new_id = self.knowledge_service.create_scene_knowledge(
            self.scene_key,
            self.shop_id,
            goods_id=self.goods_id,
            aliases=data["aliases"],
            answer=data["answer"],
            sub_intent=data["sub_intent"],
            section_title=data["section_title"],
            priority=data["priority"],
            enabled=data["enabled"],
        )
        if new_id is not None:
            self._refresh_entries()
            QMessageBox.information(self, "添加成功", f"已添加，ID={new_id}")
        else:
            QMessageBox.warning(self, "保存失败", "场景无效")

    def _refresh_entries(self):
        self.entries = self.knowledge_service.list_scene_knowledge_by_goods(
            self.scene_key,
            self.shop_id,
            self.goods_id,
        )
        self.table.setRowCount(len(self.entries))
        for row, item in enumerate(self.entries):
            values = [
                item.get("section_title", ""),
                item.get("sub_intent", ""),
                str(item.get("priority", 0)),
                item.get("aliases", ""),
                item.get("answer", ""),
            ]
            for col, value in enumerate(values):
                preview = value or ""
                if col in (3, 4) and len(preview) > 120:
                    preview = preview[:120] + "..."
                table_item = QTableWidgetItem(preview)
                if col in (0, 1, 2):
                    table_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, table_item)

            edit_btn = PushButton("编辑")
            edit_btn.clicked.connect(lambda _, r=row: self._edit_entry(r))
            self.table.setCellWidget(row, 5, edit_btn)

            delete_btn = PushButton("删除")
            delete_btn.clicked.connect(lambda _, r=row: self._delete_entry(r))
            self.table.setCellWidget(row, 6, delete_btn)

        self.detail.clear()

    def _show_entry_detail(self, row: int, column: int):
        if row < 0 or row >= len(self.entries):
            return
        item = self.entries[row]
        self.detail.setPlainText(
            "分类：{section}\n细分意图：{intent}\n优先级：{priority}\n\n问法：\n{aliases}\n\n答案：\n{answer}".format(
                section=item.get("section_title", ""),
                intent=item.get("sub_intent", ""),
                priority=item.get("priority", 0),
                aliases=item.get("aliases", ""),
                answer=item.get("answer", ""),
            )
        )

    def _edit_entry(self, row: int):
        if row < 0 or row >= len(self.entries):
            return
        entry = self.entries[row]
        dialog = SceneKnowledgeEditDialog(entry, self.window())
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        data = dialog.get_data()
        if not data["aliases"] or not data["answer"]:
            QMessageBox.warning(self, "保存失败", "问法和答案不能为空")
            return
        ok = self.knowledge_service.update_scene_knowledge(
            self.scene_key,
            entry["id"],
            aliases=data["aliases"],
            answer=data["answer"],
            sub_intent=data["sub_intent"],
            section_title=data["section_title"],
            priority=data["priority"],
            enabled=data["enabled"],
        )
        if ok:
            self._refresh_entries()
        else:
            QMessageBox.warning(self, "保存失败", "知识不存在或场景无效")

    def _delete_entry(self, row: int):
        if row < 0 or row >= len(self.entries):
            return
        entry = self.entries[row]
        confirm = QMessageBox.question(
            self, "确认删除",
            f"确定要删除这条知识吗？\n\n{entry.get('section_title', '')} › {entry.get('sub_intent', '')}\n{entry.get('answer', '')[:80]}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        ok = self.knowledge_service.delete_scene_knowledge(self.scene_key, entry["id"])
        if ok:
            self._refresh_entries()
        else:
            QMessageBox.warning(self, "删除失败", "知识不存在")


class ProductFamilySceneKnowledgeDialog(QDialog):
    """某个商品族某个场景的结构化知识列表。"""

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        shop_id: int,
        product_family: str,
        scene_key: str,
        parent=None,
    ):
        super().__init__(parent)
        self.knowledge_service = knowledge_service
        self.shop_id = shop_id
        self.product_family = product_family
        self.scene_key = scene_key
        self.entries: List[Dict] = []
        scene_label = {"presale": "售前", "insale": "售中", "aftersale": "售后"}.get(scene_key, scene_key)
        self.setWindowTitle(f"{scene_label}商品族知识 - {product_family}")
        self.resize(1180, 720)
        self.setStyleSheet(SCENE_KNOWLEDGE_LIGHT_STYLE)
        self._init_ui()
        self._refresh_entries()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"商品族：{self.product_family}"))

        self.table = TableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["分类", "细分意图", "覆盖链接", "优先级", "问法", "答案", "操作"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for col in (2, 3, 6):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 90)
        self.table.setColumnWidth(3, 70)
        self.table.setColumnWidth(6, 90)
        self.table.cellDoubleClicked.connect(lambda row, _: self._edit_entry(row))
        layout.addWidget(self.table, 1)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText("点击知识行后，这里显示完整问法和答案")
        self.table.cellClicked.connect(self._show_entry_detail)
        layout.addWidget(self.detail)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _group_entries(rows: List[Dict]) -> List[Dict]:
        grouped: Dict[tuple, Dict] = {}
        for item in rows:
            key = (
                item.get("section_title", ""),
                item.get("sub_intent", ""),
                item.get("aliases", ""),
                item.get("answer", ""),
                item.get("tags", ""),
                item.get("priority", 0),
                bool(item.get("enabled", True)),
            )
            group = grouped.setdefault(
                key,
                {
                    **item,
                    "entry_ids": [],
                    "goods_ids": set(),
                },
            )
            group["entry_ids"].append(item.get("id"))
            if item.get("goods_id") is not None:
                group["goods_ids"].add(item.get("goods_id"))

        result = []
        for item in grouped.values():
            item["goods_count"] = len(item.pop("goods_ids", set()))
            result.append(item)
        result.sort(key=lambda row: (
            str(row.get("section_title", "")),
            str(row.get("sub_intent", "")),
            -as_int(row.get("priority", 0), 0),
        ))
        return result

    def _refresh_entries(self):
        rows = self.knowledge_service.list_scene_knowledge_by_family(
            self.scene_key,
            self.shop_id,
            self.product_family,
        )
        self.entries = self._group_entries(rows)
        self.table.setRowCount(len(self.entries))
        for row, item in enumerate(self.entries):
            values = [
                item.get("section_title", ""),
                item.get("sub_intent", ""),
                str(item.get("goods_count", 0)),
                str(item.get("priority", 0)),
                item.get("aliases", ""),
                item.get("answer", ""),
            ]
            for col, value in enumerate(values):
                preview = value or ""
                if col in (4, 5) and len(preview) > 120:
                    preview = preview[:120] + "..."
                table_item = QTableWidgetItem(preview)
                if col in (0, 1, 2, 3):
                    table_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, table_item)

            edit_btn = PushButton("编辑")
            edit_btn.clicked.connect(lambda _, r=row: self._edit_entry(r))
            self.table.setCellWidget(row, 6, edit_btn)
        self.detail.clear()

    def _show_entry_detail(self, row: int, column: int):
        if row < 0 or row >= len(self.entries):
            return
        item = self.entries[row]
        self.detail.setPlainText(
            "分类：{section}\n细分意图：{intent}\n覆盖链接：{count}\n优先级：{priority}\n\n问法：\n{aliases}\n\n答案：\n{answer}".format(
                section=item.get("section_title", ""),
                intent=item.get("sub_intent", ""),
                count=item.get("goods_count", 0),
                priority=item.get("priority", 0),
                aliases=item.get("aliases", ""),
                answer=item.get("answer", ""),
            )
        )

    def _edit_entry(self, row: int):
        if row < 0 or row >= len(self.entries):
            return
        entry = self.entries[row]
        dialog = SceneKnowledgeEditDialog(entry, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        data = dialog.get_data()
        if not data["aliases"] or not data["answer"]:
            QMessageBox.warning(self, "保存失败", "问法和答案不能为空")
            return
        count = self.knowledge_service.update_scene_knowledge_entries(
            self.scene_key,
            entry.get("entry_ids", []),
            aliases=data["aliases"],
            answer=data["answer"],
            sub_intent=data["sub_intent"],
            section_title=data["section_title"],
            priority=data["priority"],
            enabled=data["enabled"],
        )
        if count:
            self._refresh_entries()
        else:
            QMessageBox.warning(self, "保存失败", "知识不存在或场景无效")


class ProductFamilyLinksDialog(QDialog):
    """商品族绑定链接列表。"""

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        shop_id: int,
        product_family: str,
        parent=None,
    ):
        super().__init__(parent)
        self.knowledge_service = knowledge_service
        self.shop_id = shop_id
        self.product_family = product_family
        self.links: List[Dict] = []
        self.setWindowTitle(f"绑定链接 - {product_family}")
        self.resize(860, 620)
        self.setStyleSheet(SCENE_KNOWLEDGE_LIGHT_STYLE)
        self._init_ui()
        self._refresh_links()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel(f"商品族：{self.product_family}"))
        toolbar.addStretch()
        self.add_link_btn = PrimaryPushButton("添加链接")
        self.add_link_btn.clicked.connect(self._add_link)
        toolbar.addWidget(self.add_link_btn)
        layout.addLayout(toolbar)

        self.table = TableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["商品ID", "商品名称", "价格"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _refresh_links(self):
        self.links = self.knowledge_service.list_product_family_links(
            self.shop_id,
            self.product_family,
        )
        self.table.setRowCount(len(self.links))
        for row, item in enumerate(self.links):
            gid = QTableWidgetItem(str(item.get("goods_id", "")))
            gid.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, gid)
            self.table.setItem(row, 1, QTableWidgetItem(item.get("goods_name", "")))
            price = QTableWidgetItem(item.get("price", ""))
            price.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, price)

    def _add_link(self):
        dialog = ProductFamilyAddLinkDialog(
            self.knowledge_service,
            self.shop_id,
            self.product_family,
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        goods_id = dialog.selected_goods_id()
        if not goods_id:
            return
        result = self.knowledge_service.bind_product_to_family(
            self.shop_id,
            self.product_family,
            goods_id,
        )
        inserted = result.get("inserted", 0)
        if inserted:
            QMessageBox.information(self, "添加成功", f"已绑定商品 {goods_id}，生成结构化知识 {inserted} 条。")
            self._refresh_links()
        else:
            QMessageBox.warning(self, "添加失败", "未生成结构化知识，请确认商品已同步且该商品族已有模板知识。")


class ProductFamilyAddLinkDialog(QDialog):
    """选择一个已同步商品绑定到商品族。"""

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        shop_id: int,
        product_family: str,
        parent=None,
    ):
        super().__init__(parent)
        self.knowledge_service = knowledge_service
        self.shop_id = shop_id
        self.product_family = product_family
        self.products: List[Dict] = []
        self._selected_goods_id: Optional[int] = None
        self.setWindowTitle(f"添加链接 - {product_family}")
        self.resize(900, 620)
        self.setStyleSheet(SCENE_KNOWLEDGE_LIGHT_STYLE)
        self._init_ui()
        self._refresh_products()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        search_bar = QHBoxLayout()
        search_bar.addWidget(QLabel("商品"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索商品ID或商品名")
        self.search_edit.textChanged.connect(self._refresh_products)
        search_bar.addWidget(self.search_edit)
        layout.addLayout(search_bar)

        self.table = TableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["商品ID", "商品名称", "价格", "操作"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(3, 90)
        self.table.cellDoubleClicked.connect(lambda row, _: self._choose(row))
        layout.addWidget(self.table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _refresh_products(self):
        products = self.knowledge_service.list_unbound_products_for_family(
            self.shop_id,
            self.product_family,
        )
        keyword = self.search_edit.text().strip().lower() if hasattr(self, "search_edit") else ""
        if keyword:
            products = [
                item for item in products
                if keyword in str(item.get("goods_id", "")).lower()
                or keyword in str(item.get("goods_name", "")).lower()
            ]
        self.products = products
        self.table.setRowCount(len(products))
        for row, item in enumerate(products):
            gid = QTableWidgetItem(str(item.get("goods_id", "")))
            gid.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, gid)
            self.table.setItem(row, 1, QTableWidgetItem(item.get("goods_name", "")))
            price = QTableWidgetItem(item.get("price", ""))
            price.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, price)
            btn = PushButton("添加")
            btn.clicked.connect(lambda _, r=row: self._choose(r))
            self.table.setCellWidget(row, 3, btn)

    def _choose(self, row: int):
        if row < 0 or row >= len(self.products):
            return
        self._selected_goods_id = int(self.products[row].get("goods_id"))
        self.accept()

    def selected_goods_id(self) -> Optional[int]:
        return self._selected_goods_id


class KnowledgeUI(QWidget):
    """知识库管理主界面"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName('KnowledgeUI')
        self.resize(900, 700)

        # 从DI容器获取服务
        self.knowledge_service: KnowledgeService = container.get(KnowledgeService)
        self.product_sync = ProductSyncService(self.knowledge_service)

        # 当前选中的店铺
        self.current_shop_id: Optional[int] = None
        # 店铺缓存 {shop_id: shop_name}
        self._shop_cache: Dict[int, str] = {}

        # 懒加载标志：只在首次切换到对应标签页时加载数据
        self._product_loaded = False
        self._cs_loaded = False
        self._meta_loaded = False
        self._scene_loaded = False
        self._family_loaded = False
        self._scene_products: List[ProductKnowledge] = []
        self._product_families: List[Dict] = []
        # 标签缓存，避免重复重建下拉框
        self._last_cs_tags: tuple = ()

        self._init_ui()
        self._load_shops()

    def _init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 顶部店铺选择栏
        shop_bar = QHBoxLayout()
        shop_bar.setSpacing(8)

        shop_label = QLabel("当前店铺:")
        shop_label.setFixedWidth(60)
        self.shop_combo = ComboBox()
        self.shop_combo.currentIndexChanged.connect(self._on_shop_changed)

        shop_bar.addWidget(shop_label)
        shop_bar.addWidget(self.shop_combo)
        shop_bar.addStretch()
        self.rebuild_index_btn = PushButton("重构知识库索引")
        self.rebuild_index_btn.clicked.connect(self._on_rebuild_index_clicked)
        shop_bar.addWidget(self.rebuild_index_btn)

        main_layout.addLayout(shop_bar)

        # SegmentedWidget 标签切换
        self.pivot = SegmentedWidget(self)
        self.pivot.setFixedWidth(420)
        self.stacked_widget = QStackedWidget(self)

        # 初始化页面
        self._init_product_tab()
        self._init_scene_tab()
        self._init_family_tab()

        # 添加页面到 stacked_widget
        self.stacked_widget.addWidget(self.product_tab)
        self.stacked_widget.addWidget(self.scene_tab)
        self.stacked_widget.addWidget(self.family_tab)

        # 添加 SegmentedWidget 按钮（带懒加载）
        self.pivot.addItem(
            routeKey="product",
            text="产品知识",
            onClick=lambda: self._switch_to_product_tab()
        )
        self.pivot.addItem(
            routeKey="scene_knowledge",
            text="场景知识库",
            onClick=lambda: self._switch_to_scene_tab()
        )
        self.pivot.addItem(
            routeKey="product_family",
            text="商品族",
            onClick=lambda: self._switch_to_family_tab()
        )
        self.pivot.setCurrentItem("product")

        # 居中放置 SegmentedWidget
        pivot_layout = QHBoxLayout()
        pivot_layout.addStretch()
        pivot_layout.addWidget(self.pivot)
        pivot_layout.addStretch()
        main_layout.addLayout(pivot_layout)

        main_layout.addWidget(self.stacked_widget)

        self.setLayout(main_layout)
        logger.info("KnowledgeUI 初始化完成")

    def _load_shops(self):
        """加载店铺列表到下拉框"""
        self.shop_combo.clear()
        self._shop_cache.clear()
        shops = self.knowledge_service.get_all_shops()
        if not shops:
            self.shop_combo.addItem("请先在账号管理添加店铺")
            self.shop_combo.setItemData(0, None)
            return

        for i, shop in enumerate(shops):
            self.shop_combo.addItem(shop.shop_name)
            self.shop_combo.setItemData(i, shop.id)
            self._shop_cache[shop.id] = shop.shop_name

        # 默认选中第一个
        if len(shops) > 0:
            self.shop_combo.setCurrentIndex(0)
            self.current_shop_id = shops[0].id
            # 懒加载：只刷新当前可见的标签页
            if self.stacked_widget.currentWidget() == self.product_tab:
                self._refresh_product_table()
                self._product_loaded = True
            elif self.stacked_widget.currentWidget() == self.family_tab:
                self._refresh_product_families()
                self._family_loaded = True
            else:
                self._refresh_scene_products()
                self._scene_loaded = True

    def _switch_to_product_tab(self):
        """切换到产品知识标签页（懒加载）"""
        self.stacked_widget.setCurrentWidget(self.product_tab)
        if self.current_shop_id is not None:
            self._refresh_product_table()
            self._product_loaded = True

    def _switch_to_cs_tab(self):
        """旧客服知识页已下线，重定向到场景知识库。"""
        self._switch_to_scene_tab()

    def _switch_to_scene_tab(self):
        """切换到场景知识库标签页。"""
        self.stacked_widget.setCurrentWidget(self.scene_tab)
        if self.current_shop_id is not None:
            self._refresh_scene_products()
            self._scene_loaded = True

    def _switch_to_family_tab(self):
        """切换到商品族标签页。"""
        self.stacked_widget.setCurrentWidget(self.family_tab)
        if self.current_shop_id is not None:
            self._refresh_product_families()
            self._family_loaded = True

    def _switch_to_meta_tab(self):
        """旧结构化知识页已下线，重定向到场景知识库。"""
        self._switch_to_scene_tab()

    def _on_shop_changed(self, index: int):
        """店铺切换（懒加载，只刷新当前可见标签页）"""
        shop_id = self.shop_combo.itemData(index)
        if shop_id is not None:
            self.current_shop_id = shop_id
            self._product_loaded = False
            self._cs_loaded = False
            self._meta_loaded = False
            self._scene_loaded = False
            self._family_loaded = False
            self._last_cs_tags = ()
            # 只刷新当前可见的标签页
            if self.stacked_widget.currentWidget() == self.product_tab:
                self._refresh_product_table()
                self._product_loaded = True
            elif self.stacked_widget.currentWidget() == self.family_tab:
                self._refresh_product_families()
                self._family_loaded = True
            else:
                self._refresh_scene_products()
                self._scene_loaded = True

    def _init_product_tab(self):
        """初始化产品知识标签页"""
        self.product_tab = QWidget()
        layout = QVBoxLayout(self.product_tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        # 顶部工具栏
        toolbar = QHBoxLayout()
        self.sync_btn = PrimaryPushButton("同步产品知识")
        self.sync_btn.clicked.connect(self._on_sync_clicked)
        self.clear_btn = PushButton("清空全部")
        self.clear_btn.clicked.connect(self._on_clear_clicked)

        toolbar.addWidget(self.sync_btn)
        toolbar.addWidget(self.clear_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.cancel_sync_btn = PushButton("取消")
        self.cancel_sync_btn.clicked.connect(self._on_cancel_sync)
        self.cancel_sync_btn.setVisible(False)

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.cancel_sync_btn)
        layout.addLayout(progress_layout)

        # 产品表格
        self.product_table = TableWidget()
        self.product_table.setColumnCount(5)
        self.product_table.setHorizontalHeaderLabels(["商品ID", "商品名称", "价格", "同步时间", "操作"])
        self.product_table.setAlternatingRowColors(True)  # 交替行颜色
        self.product_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)  # 选择整行
        self.product_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)  # 单选
        self.product_table.verticalHeader().setVisible(False)  # 隐藏行号
        self.product_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.product_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.product_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.product_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.product_table.setColumnWidth(4, 180)  # 操作列固定宽度
        self.product_table.verticalHeader().setDefaultSectionSize(50)  # 设置默认行高
        layout.addWidget(self.product_table)

    def _init_scene_tab(self):
        """初始化场景知识库标签页。"""
        self.scene_tab = QWidget()
        layout = QVBoxLayout(self.scene_tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        # ── 店铺通用知识入口 ──
        store_section = QFrame()
        store_section.setFrameShape(QFrame.Shape.StyledPanel)
        store_section.setStyleSheet("QFrame { border: 1px solid #d8dce3; border-radius: 6px; padding: 8px; background: #f7f8fa; }")
        store_layout = QVBoxLayout(store_section)
        store_layout.setContentsMargins(12, 8, 12, 8)
        store_layout.setSpacing(6)

        store_title = QLabel("店铺通用知识（对所有商品生效）")
        store_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        store_layout.addWidget(store_title)

        store_btn_layout = QHBoxLayout()
        store_btn_layout.setSpacing(8)
        for label, scene_key in (("售前知识", "presale"), ("售中知识", "insale"), ("售后知识", "aftersale")):
            btn = PushButton(label)
            btn.clicked.connect(lambda _, s=scene_key: self._open_store_scene_dialog(s))
            store_btn_layout.addWidget(btn)
        store_btn_layout.addStretch()
        store_layout.addLayout(store_btn_layout)

        layout.addWidget(store_section)

        # ── 分割线：商品级知识 ──
        sep_layout = QHBoxLayout()
        sep_line = QLabel("──────────────────── 以下为商品级知识 ────────────────────")
        sep_line.setStyleSheet("color: #999; font-size: 12px;")
        sep_layout.addWidget(sep_line)
        layout.addLayout(sep_layout)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("商品"))
        self.scene_product_search = QLineEdit()
        self.scene_product_search.setPlaceholderText("搜索商品ID或商品名")
        self.scene_product_search.textChanged.connect(self._refresh_scene_products)
        toolbar.addWidget(self.scene_product_search)
        layout.addLayout(toolbar)

        self.scene_product_table = TableWidget()
        self.scene_product_table.setColumnCount(6)
        self.scene_product_table.setHorizontalHeaderLabels(["商品ID", "商品名称", "价格", "售前", "售中", "售后"])
        self.scene_product_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.scene_product_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.scene_product_table.verticalHeader().setVisible(False)
        self.scene_product_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.scene_product_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.scene_product_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        for col in (3, 4, 5):
            self.scene_product_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.scene_product_table.setColumnWidth(col, 90)
        self.scene_product_table.setStyleSheet(SCENE_KNOWLEDGE_LIGHT_STYLE)
        layout.addWidget(self.scene_product_table)

    def _init_family_tab(self):
        """初始化商品族标签页。"""
        self.family_tab = QWidget()
        layout = QVBoxLayout(self.family_tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("商品族"))
        self.family_search = QLineEdit()
        self.family_search.setPlaceholderText("搜索商品族")
        self.family_search.textChanged.connect(self._refresh_product_families)
        toolbar.addWidget(self.family_search)
        layout.addLayout(toolbar)

        self.family_table = TableWidget()
        self.family_table.setColumnCount(8)
        self.family_table.setHorizontalHeaderLabels([
            "商品族",
            "绑定链接数",
            "售前条数",
            "售中条数",
            "售后条数",
            "售前",
            "售中",
            "售后 / 绑定链接",
        ])
        self.family_table.setAlternatingRowColors(True)
        self.family_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.family_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.family_table.verticalHeader().setVisible(False)
        self.family_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3, 4, 5, 6, 7):
            self.family_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
        self.family_table.setColumnWidth(1, 90)
        self.family_table.setColumnWidth(2, 80)
        self.family_table.setColumnWidth(3, 80)
        self.family_table.setColumnWidth(4, 80)
        self.family_table.setColumnWidth(5, 80)
        self.family_table.setColumnWidth(6, 80)
        self.family_table.setColumnWidth(7, 170)
        self.family_table.setStyleSheet(SCENE_KNOWLEDGE_LIGHT_STYLE)
        layout.addWidget(self.family_table)

    def _init_cs_tab(self):
        """初始化客服知识标签页"""
        self.cs_tab = QWidget()
        layout = QVBoxLayout(self.cs_tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        # 顶部工具栏
        toolbar = QHBoxLayout()
        self.add_cs_btn = PrimaryPushButton("添加客服知识")
        self.add_cs_btn.clicked.connect(self._on_add_cs_clicked)

        # 标签筛选
        self.tag_label = QLabel("标签筛选:")
        self.tag_combo = ComboBox()
        self.tag_combo.addItem("全部", None)
        self.tag_combo.currentIndexChanged.connect(self._on_tag_filter_changed)

        self.batch_import_btn = PushButton("批量导入")
        self.batch_import_btn.clicked.connect(self._on_batch_import_clicked)

        toolbar.addWidget(self.add_cs_btn)
        toolbar.addWidget(self.batch_import_btn)
        toolbar.addStretch()
        toolbar.addWidget(self.tag_label)
        toolbar.addWidget(self.tag_combo)
        layout.addLayout(toolbar)

        # 客服知识表格
        self.cs_table = TableWidget()
        self.cs_table.setColumnCount(6)
        self.cs_table.setHorizontalHeaderLabels(["标题", "内容", "标签", "状态", "更新时间", "操作"])
        self.cs_table.setAlternatingRowColors(True)  # 交替行颜色
        self.cs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)  # 选择整行
        self.cs_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)  # 单选
        self.cs_table.verticalHeader().setVisible(False)  # 隐藏行号
        self.cs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.cs_table.setColumnWidth(0, 160)  # 标题列固定宽度
        self.cs_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.cs_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.cs_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.cs_table.setColumnWidth(5, 180)  # 操作列固定宽度
        self.cs_table.verticalHeader().setDefaultSectionSize(50)  # 设置默认行高
        layout.addWidget(self.cs_table)

    def _init_meta_tab(self):
        """初始化结构化知识标签页"""
        self.meta_tab = QWidget()
        layout = QVBoxLayout(self.meta_tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        hint = QLabel("展示底层结构化知识：场景 / 子意图 / 问法 / 标准答案")
        toolbar.addWidget(hint)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.meta_table = TableWidget()
        self.meta_table.setColumnCount(7)
        self.meta_table.setHorizontalHeaderLabels(["来源", "商品ID", "商品族", "场景", "子意图", "问法", "标准答案"])
        self.meta_table.setAlternatingRowColors(True)
        self.meta_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.meta_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.meta_table.verticalHeader().setVisible(False)
        self.meta_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.meta_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.meta_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.meta_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.meta_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.meta_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.meta_table.verticalHeader().setDefaultSectionSize(54)
        layout.addWidget(self.meta_table)

    def _refresh_product_table(self):
        """刷新产品知识表格"""
        if self.current_shop_id is None:
            return

        products = self.knowledge_service.list_products_by_shop(self.current_shop_id)
        self.product_table.setRowCount(len(products))

        for row, product in enumerate(products):
            # 商品ID
            item = QTableWidgetItem(str(product.goods_id))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.product_table.setItem(row, 0, item)

            # 商品名称
            item = QTableWidgetItem(product.goods_name)
            self.product_table.setItem(row, 1, item)

            # 价格
            price_str = product.price or ""
            item = QTableWidgetItem(price_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.product_table.setItem(row, 2, item)

            # 同步时间
            dt_str = product.last_extracted_at.strftime("%Y-%m-%d %H:%M") if product.last_extracted_at else ""
            item = QTableWidgetItem(dt_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.product_table.setItem(row, 3, item)

            # 操作按钮 - 详情/编辑 删除
            # 使用容器放按钮
            cell_widget = QWidget()
            btn_layout = QHBoxLayout(cell_widget)
            btn_layout.setContentsMargins(4, 4, 4, 4)
            btn_layout.setSpacing(4)

            detail_btn = PushButton("详情")
            detail_btn.clicked.connect(lambda _, r=row: self._view_product(r))
            delete_btn = PushButton("删除")
            delete_btn.clicked.connect(lambda _, r=row: self._on_delete_product(r))

            btn_layout.addWidget(detail_btn)
            btn_layout.addWidget(delete_btn)
            cell_widget.setLayout(btn_layout)
            self.product_table.setCellWidget(row, 4, cell_widget)

    def _refresh_scene_products(self):
        """刷新场景知识商品列表，只加载商品摘要。"""
        if self.current_shop_id is None or not hasattr(self, "scene_product_table"):
            return

        products = self.knowledge_service.list_products_by_shop(self.current_shop_id)
        keyword = self.scene_product_search.text().strip().lower() if hasattr(self, "scene_product_search") else ""
        if keyword:
            products = [
                p for p in products
                if keyword in str(p.goods_id).lower() or keyword in str(p.goods_name or "").lower()
            ]
        self._scene_products = products
        self.scene_product_table.setRowCount(len(products))
        for row, product in enumerate(products):
            gid_item = QTableWidgetItem(str(product.goods_id))
            gid_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.scene_product_table.setItem(row, 0, gid_item)
            self.scene_product_table.setItem(row, 1, QTableWidgetItem(product.goods_name or ""))
            price_item = QTableWidgetItem(product.price or "")
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.scene_product_table.setItem(row, 2, price_item)

            for col, (label, scene_key) in enumerate(
                (("售前", "presale"), ("售中", "insale"), ("售后", "aftersale")),
                start=3,
            ):
                btn = PushButton(label)
                btn.clicked.connect(lambda _, r=row, s=scene_key: self._open_scene_dialog(r, s))
                self.scene_product_table.setCellWidget(row, col, btn)

    def _refresh_product_families(self):
        """刷新商品族列表。"""
        if self.current_shop_id is None or not hasattr(self, "family_table"):
            return

        families = self.knowledge_service.list_product_families_by_shop(self.current_shop_id)
        keyword = self.family_search.text().strip().lower() if hasattr(self, "family_search") else ""
        if keyword:
            families = [
                item for item in families
                if keyword in str(item.get("product_family", "")).lower()
            ]
        self._product_families = families
        self.family_table.setRowCount(len(families))
        for row, item in enumerate(families):
            values = [
                item.get("product_family", ""),
                str(item.get("goods_count", 0)),
                str(item.get("presale_count", 0)),
                str(item.get("insale_count", 0)),
                str(item.get("aftersale_count", 0)),
            ]
            for col, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                if col > 0:
                    table_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.family_table.setItem(row, col, table_item)

            for col, (label, scene_key) in enumerate(
                (("售前", "presale"), ("售中", "insale")),
                start=5,
            ):
                btn = PushButton(label)
                btn.clicked.connect(lambda _, r=row, s=scene_key: self._open_family_scene_dialog(r, s))
                self.family_table.setCellWidget(row, col, btn)

            cell_widget = QWidget()
            btn_layout = QHBoxLayout(cell_widget)
            btn_layout.setContentsMargins(4, 4, 4, 4)
            btn_layout.setSpacing(4)
            aftersale_btn = PushButton("售后")
            aftersale_btn.clicked.connect(lambda _, r=row: self._open_family_scene_dialog(r, "aftersale"))
            links_btn = PushButton("绑定链接")
            links_btn.clicked.connect(lambda _, r=row: self._open_family_links_dialog(r))
            btn_layout.addWidget(aftersale_btn)
            btn_layout.addWidget(links_btn)
            cell_widget.setLayout(btn_layout)
            self.family_table.setCellWidget(row, 7, cell_widget)

    def _open_store_scene_dialog(self, scene_key: str):
        """打开店铺通用知识的场景管理。"""
        dialog = SceneKnowledgeListDialog(
            self.knowledge_service,
            self.current_shop_id,
            None,
            "店铺通用",
            scene_key,
            self,
        )
        dialog.exec()

    def _open_scene_dialog(self, row: int, scene_key: str):
        """打开某个商品某个场景的知识明细。"""
        if row < 0 or row >= len(self._scene_products):
            return
        product = self._scene_products[row]
        dialog = SceneKnowledgeListDialog(
            self.knowledge_service,
            self.current_shop_id,
            int(product.goods_id),
            product.goods_name or "",
            scene_key,
            self,
        )
        dialog.exec()

    def _open_family_scene_dialog(self, row: int, scene_key: str):
        """打开某个商品族某个场景的知识明细。"""
        if row < 0 or row >= len(self._product_families):
            return
        product_family = str(self._product_families[row].get("product_family") or "")
        dialog = ProductFamilySceneKnowledgeDialog(
            self.knowledge_service,
            self.current_shop_id,
            product_family,
            scene_key,
            self,
        )
        dialog.exec()
        self._refresh_product_families()

    def _open_family_links_dialog(self, row: int):
        """打开某个商品族绑定链接列表。"""
        if row < 0 or row >= len(self._product_families):
            return
        product_family = str(self._product_families[row].get("product_family") or "")
        dialog = ProductFamilyLinksDialog(
            self.knowledge_service,
            self.current_shop_id,
            product_family,
            self,
        )
        dialog.exec()

    def _refresh_cs_table(self):
        """刷新客服知识表格"""
        if self.current_shop_id is None:
            return

        current_selection = self.tag_combo.currentData()

        # 更新标签下拉框（只有标签变化时才重建，避免卡顿）
        all_tags = tuple(sorted(self.knowledge_service.get_all_tags(self.current_shop_id)))
        if all_tags != self._last_cs_tags:
            self._last_cs_tags = all_tags

            self.tag_combo.blockSignals(True)
            self.tag_combo.clear()
            self.tag_combo.addItem("全部")
            self.tag_combo.setItemData(0, None)
            for i, tag in enumerate(all_tags, 1):
                self.tag_combo.addItem(tag)
                self.tag_combo.setItemData(i, tag)
            # 恢复选中
            if current_selection is None:
                self.tag_combo.setCurrentIndex(0)
            else:
                # 查找索引
                for i in range(self.tag_combo.count()):
                    if self.tag_combo.itemData(i) == current_selection:
                        self.tag_combo.setCurrentIndex(i)
                        break
            self.tag_combo.blockSignals(False)
            current_selection = self.tag_combo.currentData()

        # 获取数据
        if current_selection is None:
            cs_list = self.knowledge_service.list_customer_service_with_disabled(self.current_shop_id)
        else:
            cs_list = self.knowledge_service.filter_customer_service_by_tag(self.current_shop_id, current_selection)

        self.cs_table.setRowCount(len(cs_list))

        for row, cs in enumerate(cs_list):
            # 标题
            item = QTableWidgetItem(cs.title)
            self.cs_table.setItem(row, 0, item)

            # 内容（截断避免过长）
            content_preview = cs.content
            if len(content_preview) > 60:
                content_preview = content_preview[:60] + "..."
            item = QTableWidgetItem(content_preview)
            item.setToolTip(cs.content)
            self.cs_table.setItem(row, 1, item)

            # 标签
            item = QTableWidgetItem(cs.tags or "")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.cs_table.setItem(row, 2, item)

            # 状态
            status_text = "启用" if cs.enabled else "禁用"
            item = QTableWidgetItem(status_text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.cs_table.setItem(row, 3, item)

            # 更新时间
            dt_str = cs.updated_at.strftime("%Y-%m-%d %H:%M") if cs.updated_at else ""
            item = QTableWidgetItem(dt_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.cs_table.setItem(row, 4, item)

            # 操作按钮
            cell_widget = QWidget()
            btn_layout = QHBoxLayout(cell_widget)
            btn_layout.setContentsMargins(4, 4, 4, 4)
            btn_layout.setSpacing(4)

            edit_btn = PushButton("编辑")
            edit_btn.clicked.connect(lambda _, r=row: self._on_edit_cs(r))
            delete_btn = PushButton("删除")
            delete_btn.clicked.connect(lambda _, r=row: self._on_delete_cs(r))

            btn_layout.addWidget(edit_btn)
            btn_layout.addWidget(delete_btn)
            cell_widget.setLayout(btn_layout)
            self.cs_table.setCellWidget(row, 5, cell_widget)

            # 禁用行灰色显示
            if not cs.enabled:
                for col in range(self.cs_table.columnCount()):
                    if self.cs_table.item(row, col):
                        self.cs_table.item(row, col).setForeground(Qt.GlobalColor.gray)

    def _refresh_meta_table(self):
        """刷新结构化知识表格"""
        if self.current_shop_id is None:
            return

        with self.knowledge_service.get_session() as session:
            from sqlalchemy import select
            stmt = select(KnowledgeMetaEntry).where(
                KnowledgeMetaEntry.shop_id == self.current_shop_id
            ).order_by(
                KnowledgeMetaEntry.updated_at.desc(),
                KnowledgeMetaEntry.id.desc(),
            )
            meta_list = list(session.scalars(stmt))

        self.meta_table.setRowCount(len(meta_list))
        for row, meta in enumerate(meta_list):
            source_text = "商品" if meta.source_type == "product" else "客服"
            values = [
                source_text,
                str(meta.goods_id or ""),
                meta.product_family or "",
                meta.scenario or "",
                meta.sub_intent or "",
                meta.aliases or "",
                meta.answer or "",
            ]
            for col, value in enumerate(values):
                preview = value
                if col in (5, 6) and len(preview) > 80:
                    preview = preview[:80] + "..."
                item = QTableWidgetItem(preview)
                item.setToolTip(value)
                if col <= 4:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.meta_table.setItem(row, col, item)

    def _view_product(self, row: int):
        """查看/编辑产品详情"""
        # 获取商品ID
        product_id = self.product_table.item(row, 0).text()
        goods_id = int(product_id)
        product = self.knowledge_service.get_product_by_goods_id(self.current_shop_id, goods_id)
        if not product:
            QMessageBox.warning(self, "错误", "产品不存在")
            return

        dialog = ProductDetailDialog(product, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            # 更新
            if product.extracted_content != data["extracted_content"] or product.goods_name != data["goods_name"]:
                with self.knowledge_service.get_session() as session:
                    prod = session.get(ProductKnowledge, product.id)
                    if prod:
                        prod.goods_name = data["goods_name"]
                        prod.extracted_content = data["extracted_content"]
                        session.commit()
                        self._show_message("success", "更新成功")
                        self._refresh_product_table()

    def _on_delete_product(self, row: int):
        """删除产品"""
        # 获取商品ID（第0列）
        product_id = self.product_table.item(row, 0).text()
        goods_id = int(product_id)
        product = self.knowledge_service.get_product_by_goods_id(self.current_shop_id, goods_id)
        if not product:
            QMessageBox.warning(self, "错误", "产品不存在")
            return

        confirm = QMessageBox.question(
            self, "确认删除",
            f"确定要删除产品 «{product.goods_name}» 吗？\n\n删除后无法恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            success = self.knowledge_service.delete_product(product.id)
            if success:
                self._show_message("success", "删除成功")
                self._refresh_product_table()
            else:
                self._show_message("error", "删除失败")

    def _on_clear_clicked(self):
        """清空全部产品知识"""
        if self.current_shop_id is None:
            return

        confirm = QMessageBox.question(
            self, "确认清空",
            f"确定要清空当前店铺的所有产品知识吗？\n\n清空后无法恢复，请谨慎操作。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            total_deleted = self.knowledge_service.clear_products_by_shop(self.current_shop_id)
            self._show_message("success", f"已清空，共删除 {total_deleted} 条记录")
            self._refresh_product_table()

    def _on_sync_clicked(self):
        """点击同步按钮，弹出选择同步模式"""
        if self.current_shop_id is None:
            self._show_message("warning", "请先选择店铺")
            return

        # 获取当前选中的店铺
        shop_to_sync = self._get_shop_by_id(self.current_shop_id)
        if not shop_to_sync:
            self._show_message("error", "无法获取店铺信息")
            return

        # 弹出选择同步模式对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择同步模式")
        dialog.resize(350, 180)

        layout = QVBoxLayout(dialog)

        label = QLabel(f"即将同步店铺 «{shop_to_sync.shop_name}» 的产品知识，请选择同步模式:")
        layout.addWidget(label)

        incremental_btn = PushButton("增量同步（仅同步本地不存在的商品，推荐）")
        full_btn = PrimaryPushButton("全量同步（同步所有商品，覆盖已提取知识）")

        layout.addWidget(incremental_btn)
        layout.addWidget(full_btn)

        def start_sync(is_full):
            dialog.close()
            self._start_sync(shop_to_sync, is_full)

        incremental_btn.clicked.connect(lambda: start_sync(False))
        full_btn.clicked.connect(lambda: start_sync(True))

        dialog.setLayout(layout)
        dialog.exec()

    def _get_shop_by_id(self, shop_id: int) -> Optional[Shop]:
        """根据ID获取店铺对象"""
        # 根据ID查询店铺对象，同时预加载关联的accounts，避免懒加载问题
        with self.knowledge_service.get_session() as session:
            from sqlalchemy import select
            from sqlalchemy.orm import joinedload
            stmt = select(Shop).where(Shop.id == shop_id).options(joinedload(Shop.accounts))
            return session.scalar(stmt)

    def _start_sync(self, shop: Shop, is_full_sync: bool):
        """开始同步"""
        # 获取pdd shop_id和user_id
        pdd_shop_id = shop.shop_id
        # 从shop.accounts[0]获取user_id，假设一个店铺只有一个账号
        if not shop.accounts:
            self._show_message("error", "店铺没有账号信息")
            return

        user_id = shop.accounts[0].user_id

        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.cancel_sync_btn.setVisible(True)
        self.sync_btn.setEnabled(False)

        # 创建工作线程
        self._sync_worker = SyncWorker(
            shop_db_id=shop.id,
            pdd_shop_id=pdd_shop_id,
            user_id=user_id,
            is_full_sync=is_full_sync,
            product_sync=self.product_sync,
            parent=self,
        )

        # 连接信号
        def on_progress(current: int, total: int, success: int, current_name: str, phase: str):
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            # 根据阶段显示不同的提示
            if phase == "fetching":
                self.progress_label.setText(f"[1/2] 抓取商品列表: {current_name} ({current}/{total})")
            elif phase == "saving_basic":
                self.progress_label.setText(f"[2/2] 保存商品信息: {current_name} ({current}/{total}, 成功 {success})")
                # 保存阶段定期刷新表格，让用户能看到新同步出来的商品。
                if current == 1 or current % 10 == 0:
                    self._refresh_product_table()
            elif phase == "extracting":
                self.progress_label.setText(f"[3/3] 提取商品知识: {current_name} ({current}/{total}, 成功 {success})")
                # 提取阶段也定期刷新，显示更新的知识
                if current % 5 == 0:
                    self._refresh_product_table()
            else:
                self.progress_label.setText(f"正在同步: {current_name} ({current}/{total}, 成功 {success})")

        def on_finished(success: int, failed: int, cancelled: bool):
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)
            self.cancel_sync_btn.setVisible(False)
            self.sync_btn.setEnabled(True)

            # 最后刷新一次表格
            self._refresh_product_table()

            if cancelled:
                self._show_message("info", "同步已取消")
            else:
                msg = f"同步完成: 成功 {success}, 失败 {failed}"
                self._show_message("success", msg)

        self._sync_worker.progress_updated.connect(on_progress)
        self._sync_worker.sync_finished.connect(on_finished)
        self._sync_worker.start()

    def _on_cancel_sync(self):
        """取消同步"""
        if hasattr(self, '_sync_worker') and self._sync_worker.isRunning():
            self.product_sync.cancel()
            self.cancel_sync_btn.setEnabled(False)

    def _on_rebuild_index_clicked(self):
        """一键重构当前店铺知识库索引。"""
        if self.current_shop_id is None:
            self._show_message("warning", "请先选择店铺")
            return
        if hasattr(self, "_index_worker") and self._index_worker.isRunning():
            self._show_message("info", "知识库索引正在重构中")
            return

        confirm = QMessageBox.question(
            self,
            "确认重构索引",
            "将清理当前店铺旧的场景知识向量索引并重新构建。过程可能需要一段时间，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self.rebuild_index_btn.setEnabled(False)
        self.rebuild_index_btn.setText("重构中...")
        self._index_worker = IndexRebuildWorker(
            self.knowledge_service,
            self.current_shop_id,
            self,
        )

        def on_finished(success: bool, message: str):
            self.rebuild_index_btn.setEnabled(True)
            self.rebuild_index_btn.setText("重构知识库索引")
            if success:
                self._show_message("success", f"索引重构完成：{message}")
            else:
                self._show_message("error", f"索引重构失败：{message}")

        self._index_worker.rebuild_finished.connect(on_finished)
        self._index_worker.start()

    def _on_add_cs_clicked(self):
        """添加客服知识"""
        if self.current_shop_id is None:
            self._show_message("warning", "请先选择店铺")
            return

        dialog = CsAddEditDialog(self.current_shop_id, None, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            self.knowledge_service.add_customer_service(
                shop_id=self.current_shop_id,
                title=data["title"],
                content=data["content"],
                tags=data["tags"],
                enabled=data["enabled"],
            )
            self._show_message("success", "添加成功")
            self._refresh_cs_table()

    def _on_batch_import_clicked(self):
        """批量导入客服知识"""
        if self.current_shop_id is None:
            self._show_message("warning", "请先选择店铺")
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "选择客服话术文件",
            "",
            "Excel 文件 (*.xls *.xlsx)",
        )
        if not filepath:
            return

        try:
            rows, parse_skipped = self._parse_excel(filepath)
        except Exception as e:
            self._show_message("error", f"文件读取失败: {e}")
            return

        success, import_skipped = self.knowledge_service.batch_import_customer_service(
            self.current_shop_id, rows
        )
        total_skipped = parse_skipped + import_skipped
        self._show_message("success", f"导入完成：成功 {success} 条，跳过 {total_skipped} 条")
        self._refresh_cs_table()

    def _parse_excel(self, filepath: str) -> tuple[list, int]:
        """解析 Excel 文件，返回 (有效行列表, 跳过行数)

        列顺序：0=一级分类, 1=二级分类, 2=话术标题, 3=话术内容
        """
        import pandas as pd

        df = pd.read_excel(filepath, header=0, dtype=str)
        df = df.fillna("")

        rows = []
        skipped = 0
        for _, row in df.iterrows():
            values = row.tolist()
            # 补齐不足4列的情况
            while len(values) < 4:
                values.append("")

            cat1 = str(values[0]).strip()
            cat2 = str(values[1]).strip()
            title = str(values[2]).strip()
            content = str(values[3]).strip()

            if not cat1 or not content:
                skipped += 1
                continue

            tags = f"{cat1},{cat2}" if cat2 else cat1
            rows.append({"title": title, "content": content, "tags": tags})

        return rows, skipped

    def _on_edit_cs(self, row: int):
        """编辑客服知识"""
        cs_id = self._get_cs_id_from_row(row)
        cs = self.knowledge_service.get_customer_service_by_id(cs_id)
        if not cs:
            self._show_message("error", "知识不存在")
            return

        dialog = CsAddEditDialog(cs.shop_id, cs, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            updated = self.knowledge_service.update_customer_service(
                cs_id,
                title=data["title"],
                content=data["content"],
                tags=data["tags"],
                enabled=data["enabled"],
            )
            if updated:
                self._show_message("success", "更新成功")
                self._refresh_cs_table()

    def _on_delete_cs(self, row: int):
        """删除客服知识"""
        cs_id = self._get_cs_id_from_row(row)
        cs = self.knowledge_service.get_customer_service_by_id(cs_id)
        if not cs:
            self._show_message("error", "知识不存在")
            return

        confirm = QMessageBox.question(
            self, "确认删除",
            f"确定要删除客服知识 «{cs.title}» 吗？\n\n删除后无法恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            success = self.knowledge_service.delete_customer_service(cs_id)
            if success:
                self._show_message("success", "删除成功")
                self._refresh_cs_table()
            else:
                self._show_message("error", "删除失败")

    def _get_cs_id_from_row(self, row: int) -> int:
        """从表格行获取客服知识ID，这里需要查询，因为表格没有保存id"""
        # 标题在第0列
        title = self.cs_table.item(row, 0).text()
        # 直接查询当前店铺下的客服知识
        with self.knowledge_service.get_session() as session:
            from sqlalchemy import select
            stmt = select(CustomerServiceKnowledge).where(
                CustomerServiceKnowledge.shop_id == self.current_shop_id,
                CustomerServiceKnowledge.title == title,
            )
            cs = session.scalar(stmt)
            if cs:
                return cs.id
        return 0

    def _on_tag_filter_changed(self, index: int):
        """标签筛选变化"""
        self._refresh_cs_table()

    def _show_message(self, level: str, content: str):
        """显示消息条"""
        method = getattr(InfoBar, level)
        method(
            title="",
            content=content,
            orient=InfoBarPosition.TOP,
            parent=self,
        )

    def showEvent(self, event):
        """显示时刷新"""
        super().showEvent(event)
        # 刷新店铺列表，可能有新增
        self._load_shops()
        if self.current_shop_id is not None:
            if self.stacked_widget.currentWidget() == self.product_tab:
                self._refresh_product_table()
                self._product_loaded = True
            elif self.stacked_widget.currentWidget() == self.family_tab:
                self._refresh_product_families()
                self._family_loaded = True
            else:
                self._refresh_scene_products()
                self._scene_loaded = True
