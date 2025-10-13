import os
import json
import base64
import tempfile
from typing import Optional, List
import requests


def _encode_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # default use jpeg data url
    return f"data:image/jpeg;base64,{b64}"


# Inline prompt templates from the provided file (at least the one we use)
PROMPT_TEMPLATES = {
    "prompt": (
        "Given an image of {} with partial occlusion, generate a concise one-sentence inpainting prompt "
        "that directly describes the {} overall shape, color, and texture for restoration, without mentioning "
        "any occlusion details or background information."
    ),
    # other templates from the file can be added if needed:
    "occluding_object": (
        'Given an image, perform the following analysis for the instance of {}: Identify Occlusions: Examine {} and determine if any portions of it are occluded by other objects in the image. If occlusions exist, list the names of the objects that are directly blocking {}. Fixed Format Output: Present the results using the following fixed format (for each {} analyzed): [<comma-separated list of names of objects occluding {}>] Ensure that your output strictly adheres to the fixed format provided above.'
    ),
    "boundary": (
    'Review the image to determine whether the entire {} is visible within its boundaries. If any part of the {} is cut off, specify the affected edge(s) (top, bottom, left, or right). For each impacted edge, provide an estimate of the missing portion as a relative fraction of the overall dimension (for example, "extends upward by 0.25 of the image height"), and indicate the necessary expansion. Output your analysis as a JSON object following the structure below. Ensure that the "extension_amount" values are expressed as numeric relative proportions (e.g., 0.25 for one quarter), not as strings: {{"is_occluded_object_outside": <boolean>, "extension_direction":  [""], "extension_amount": number}}'
    ),
    "boundary_bbox": (
    "Review the image to determine whether the entire {} is visible within its boundaries.\n\n"
    "Preliminary Analysis:\n"
    "-------------------------------------------------\n"
    "{{\n"
    "  \"exceeded_edges\": [<list of edges that have been detected as exceeded>]\n"
    "}}\n"
    "-------------------------------------------------\n\n"
        "Note: The preliminary analysis above indicates which edge(s) the object's bounding box has exceeded. "
        "For any edge listed in \"exceeded_edges\", it is confirmed that the object extends beyond the image boundary on that side. "
        "For other cases—such as when the object is partially occluded by overlapping elements—rely solely on your visual analysis.\n\n"
        "If any part of the {} is cut off, specify the affected edge(s) (top, bottom, left, or right). For each impacted edge, provide "
        "an estimate of the missing portion as a relative fraction of the corresponding dimension (for example, \"extends upward by 0.25 of the image height\"), "
        "and indicate the necessary expansion.\n\n"
        "Output your analysis as a JSON object following the structure below. Ensure that the \"extension_amount\" values are expressed as numeric "
        "relative proportions (e.g., 0.25 for one quarter), not as strings:\n\n"
    "{{\"is_occluded_object_outside\": <boolean>, \"extension_direction\": [list of affected edges], \"extension_amount\": number}}"
    ),
}


class GPTAdapter:
    """Self-contained adapter that builds prompts and (optionally) calls OpenRouter API without importing external files."""

    def __init__(
        self,
        backend: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
    ):
        # backends: 'openrouter' (default) | 'azure'
        self.backend = (backend or os.getenv("GPT_BACKEND", "openrouter")).lower()
        # OpenRouter settings
        self.or_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.or_api_key = os.getenv("OPENROUTER_API_KEY", "")
        # Default models (can be overridden via env):
        # General agents: OPENROUTER_MODEL_GENERAL or OPENROUTER_MODEL or fallback to GPT-4o
        self.model_name_general = os.getenv(
            "OPENROUTER_MODEL_GENERAL",
            os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-2024-11-20"),
        )
    # No checker agent

        # Azure OpenAI settings
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.azure_api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        # For Azure, you must pass the deployment name (not the model id). Allow override via env.
        # Fallback to a generic name if not provided.
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("AZURE_OPENAI_MODEL", "gpt-4o"))

    # --- helpers for boundary_bbox ---
    @staticmethod
    def get_exceeded_edges(bbox_json_path: str, occluded_object: str) -> List[str]:
        """Read detections json and check which edges are exceeded for `<occluded_object>_0`."""
        try:
            with open(bbox_json_path, "r") as f:
                data = json.load(f)
        except Exception:
            return []
        key = f"{occluded_object}_0"
        if key not in data:
            return []
        try:
            bbox = data[key]["bounding_box_xyxy"]
            image_width = data[key]["image_width"]
            image_height = data[key]["image_height"]
        except Exception:
            return []
        thr = 10
        exceeded = []
        if bbox[0] <= thr:
            exceeded.append("left")
        if bbox[1] <= thr:
            exceeded.append("top")
        if bbox[2] >= (image_width - thr):
            exceeded.append("right")
        if bbox[3] >= (image_height - thr):
            exceeded.append("bottom")
        return exceeded

    def _build_prompt(self, prompt_type: str, seg_text: str, bbox_json_path: Optional[str] = None, occluded_object: Optional[str] = None) -> str:
        template = PROMPT_TEMPLATES.get(prompt_type) or PROMPT_TEMPLATES["prompt"]
        # Support both positional {} and named {object_name} placeholders (do not break literal braces in examples)
        prompt_text = template
        if "{object_name}" in prompt_text:
            prompt_text = prompt_text.replace("{object_name}", seg_text)
        else:
            # substitute {} with seg_text across occurrences
            count = prompt_text.count("{}")
            if count:
                prompt_text = prompt_text.format(*([seg_text] * count))
        # boundary_bbox needs exceeded_edges injection
        if prompt_type == "boundary_bbox":
            edges: List[str] = []
            if bbox_json_path and occluded_object:
                edges = self.get_exceeded_edges(bbox_json_path, occluded_object)
            prompt_text = prompt_text.replace("[<list of edges that have been detected as exceeded>]", json.dumps(edges))
        return prompt_text

    def _openrouter_chat_with_image(self, image_path: str, system_prompt: str, model_override: Optional[str] = None) -> Optional[str]:
        if not self.or_api_key:
            return None
        url = f"{self.or_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.or_api_key}",
            "Content-Type": "application/json",
        }
        try:
            data_url = _encode_image_to_data_url(image_path)
            payload = {
                "model": model_override or self.model_name_general,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code != 200:
                return None
            obj = resp.json()
            return obj.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:
            return None

    # --- Azure backend ---
    def _azure_chat_with_image(self, image_path: str, system_prompt: str) -> Optional[str]:
        """Call Azure OpenAI Chat Completions with vision input.

        Priority: try official SDK (openai.AzureOpenAI). If unavailable, fallback to REST.
        Expects envs AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT.
        """
        key = self.azure_api_key
        ep = (self.azure_endpoint or "").rstrip("/")
        ver = self.azure_api_version
        dep = self.azure_deployment
        if not key or not ep or not dep:
            return None

        # Build messages with data URL image
        try:
            data_url = _encode_image_to_data_url(image_path)
        except Exception:
            data_url = None
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    [
                        {"type": "text", "text": system_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                    if data_url
                    else system_prompt
                ),
            },
        ]

        # Try SDK first
        try:
            try:
                from openai import AzureOpenAI  # type: ignore
            except Exception:
                AzureOpenAI = None  # type: ignore
            if AzureOpenAI is not None:
                client = AzureOpenAI(api_key=key, azure_endpoint=ep, api_version=ver)
                resp = client.chat.completions.create(
                    model=dep,
                    messages=messages,
                    temperature=0.2,
                    top_p=1,
                )
                # SDK object -> text
                try:
                    return resp.choices[0].message.content  # type: ignore
                except Exception:
                    pass
        except Exception:
            # fall through to REST
            pass

        # REST fallback
        try:
            url = f"{ep}/openai/deployments/{dep}/chat/completions?api-version={ver}"
            headers = {"api-key": key, "Content-Type": "application/json"}
            payload = {
                "messages": messages,
                "temperature": 0.2,
                "top_p": 1,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:
            return None

    def inpaint_prompt(self, seg_text: str) -> str:
        """Build prompt text strictly from the provided file if it exposes templates; otherwise minimal fallback."""
        template = PROMPT_TEMPLATES.get("prompt")
        count = template.count("{}")
        return template.format(*([seg_text] * count))

    def gen_inpaint_prompt_from_image(self, image, seg_text: str, prompt_type: str = "prompt", bbox_json_path: Optional[str] = None, occluded_object: Optional[str] = None) -> str:
        """Use the provided module's api_response with its exact prompt template, passing the image file path."""
        # Build prompt by template (purely from inline templates)
        prompt_text = self._build_prompt(prompt_type, seg_text, bbox_json_path=bbox_json_path, occluded_object=occluded_object)
        # Mock backend: return deterministic outputs without calling any external APIs
        if self.backend == "mock":
            # default mock: just return the single-sentence description
            return self.inpaint_prompt(seg_text)
        # Save image to temp file and call api_response
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tf:
            # Try saving using the object API; if not available, assume it's already bytes
            try:
                image.save(tf.name)
            except Exception:
                try:
                    tf.write(image)
                    tf.flush()
                except Exception:
                    # fall back to text-only prompt
                    return prompt_text
            # Dispatch by backend
            if self.backend == "azure":
                resp_text = self._azure_chat_with_image(tf.name, prompt_text)
            else:
                resp_text = self._openrouter_chat_with_image(tf.name, prompt_text, model_override=None)
            if resp_text:
                return resp_text
        # If cannot call api_response, return prompt text
        return prompt_text

    def generate_inpaint_description(self, seg_text: str, **kwargs) -> str:
        """Prefer calling a dedicated function in the external module that handles both prompt building and GPT call."""
        # For compatibility: just return the inline template prompt
        return self.inpaint_prompt(seg_text)

    # Checker-related helpers and prompts removed
