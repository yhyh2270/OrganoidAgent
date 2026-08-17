import argparse
import datetime
import json
import os
import uuid

import tornado.ioloop
import tornado.web
import tornado.escape

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

OPTIONS = {
    "eyes": [
        {"id": "nanobanana", "label": "Nano Banana"},
        {"id": "mistral-vision", "label": "Mistral 31 Vision"},
    ],
    "minds": [
        {"id": "chatgpt-5.2", "label": "ChatGPT 5.2"},
        {"id": "deepseek", "label": "DeepSeek"},
    ],
    "hands": [
        {"id": "codex-sdk", "label": "Codex SDK"},
        {"id": "copilot-sdk", "label": "Copilot SDK"},
    ],
    "tasks": [
        {
            "id": "segmentation",
            "label": "Segmentation",
            "prompt": "Segment organoids and immune cells; return masks and QC metrics.",
        },
        {
            "id": "tracking",
            "label": "Tracking",
            "prompt": "Track immune cell trajectories and summarize motility metrics.",
        },
        {
            "id": "composition",
            "label": "Cell-Type Composition",
            "prompt": "Estimate epithelial subtype composition from brightfield morphology.",
        },
        {
            "id": "stats",
            "label": "Statistics",
            "prompt": "Summarize experiment-level statistics and significance tests.",
        },
    ],
    "defaults": {
        "eyes": "nanobanana",
        "mind": "chatgpt-5.2",
        "hand": "codex-sdk",
        "task": "segmentation",
    },
}

JOBS = []


class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")

    def options(self, *_args, **_kwargs):
        self.set_status(204)
        self.finish()


class HealthHandler(BaseHandler):
    def get(self):
        self.write({"status": "ok"})


class OptionsHandler(BaseHandler):
    def get(self):
        self.write(OPTIONS)


class RunHandler(BaseHandler):
    def post(self):
        payload = tornado.escape.json_decode(self.request.body or b"{}")
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "status": "queued",
            "eyes": payload.get("eyes"),
            "mind": payload.get("mind"),
            "hand": payload.get("hand"),
            "task": payload.get("task"),
            "prompt": payload.get("prompt", "").strip(),
            "notes": payload.get("notes", "").strip(),
            "inputs": payload.get("inputs", []),
            "summary": "Job queued. Configure hand drivers to execute analysis.",
        }
        JOBS.append(job)
        self.write(job)


class JobsHandler(BaseHandler):
    def get(self):
        self.write({"jobs": JOBS[-50:]})


def make_app():
    return tornado.web.Application(
        [
            (r"/api/health", HealthHandler),
            (r"/api/options", OptionsHandler),
            (r"/api/run", RunHandler),
            (r"/api/jobs", JobsHandler),
            (
                r"/(.*)",
                tornado.web.StaticFileHandler,
                {"path": WEB_DIR, "default_filename": "index.html"},
            ),
        ],
        debug=True,
    )


def main():
    parser = argparse.ArgumentParser(description="BioAgent Tornado app")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    app = make_app()
    app.listen(args.port)
    print(f"BioAgent backend listening on port {args.port}")
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
