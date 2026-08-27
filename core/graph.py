import time
import uuid


class GraphTracker:
    def __init__(self):
        self.nodes = []
        self.current_parent = None

    def add_node(self, node_type: str, content: any):
        node_id = str(uuid.uuid4())
        node = {
            "id": node_id,
            "type": node_type,
            "content": content,
            "timestamp": time.time(),
            "parent_id": self.current_parent
        }
        self.nodes.append(node)
        self.current_parent = node_id
        return node_id