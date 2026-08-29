import time
import uuid

from pydantic import BaseModel


class GraphTracker:
    def __init__(self):
        self.nodes = []
        self.current_parent = None

    def add_node(self, node_type: str, content: any, parent_override:str = None) -> str:
        node_id = str(uuid.uuid4())
        node = {
            "id": node_id,
            "type": node_type,
            "content": content,
            "timestamp": time.time(),
            "parent_id": parent_override or self.current_parent
        }
        self.nodes.append(node)
        self.current_parent = node_id
        return node_id

class ReplayOverrideRequest(BaseModel):
    node_id: str
    override_result: str