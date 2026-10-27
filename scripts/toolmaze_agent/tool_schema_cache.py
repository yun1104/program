"""Cache immutable YAML schemas, with independent copies for each environment."""
from copy import deepcopy
from pathlib import Path


def enable_schema_cache(upstream):
    from tools.loader import ToolLoader
    from .common import fingerprint
    if getattr(ToolLoader, "_project_schema_cache", False):
        return
    directory = (Path(upstream) / "tools/definitions").resolve()
    prototype = ToolLoader(str(directory))
    cached = deepcopy((prototype.tools_by_category, prototype.tools_by_name))
    original = ToolLoader._load_all

    def load(self):
        if Path(self.definitions_dir).resolve() == directory:
            self.tools_by_category, self.tools_by_name = deepcopy(cached)
        else:
            original(self)

    ToolLoader._load_all = load
    ToolLoader._project_schema_cache = True
    check = ToolLoader(str(directory))
    if check.tools_by_name != prototype.tools_by_name or check.tools_by_category != prototype.tools_by_category:
        raise RuntimeError("Schema cache changed tool definitions")
    print("Independent-copy schema cache enabled:", fingerprint(check.tools_by_name), flush=True)
