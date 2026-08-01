
import argparse
import json
import pathlib
import re
import time
from PIL import Image

PROMPT = """You are an external VLM-CoT observer for a CARLA autonomous-driving demo.
Look only at the camera image and reason conservatively about the driving scene.
You must NOT output steering, throttle, brake, or any low-level control.
Return ONLY valid JSON with these keys:
{
  "scene_type": "normal|blocked_lane|traffic_light|pedestrian|vehicle_cut_in|traffic_jam|unknown",
  "risk_level": "low|medium|high|critical|unknown",
  "main_hazard": "short phrase",
  "safe_hint": "short semantic hint for the driving stack",
  "reason": "one concise natural-language reasoning sentence",
  "confidence": 0.0
}
Critical rules:
- Be conservative. If you are unsure, use risk_level "medium" and confidence <= 0.55.
- Do NOT say "normal" or "low" if there is any vehicle close ahead, beside the ego, partially blocking the lane, merging, stopped, crashed, or overlapping another vehicle.
- A close vehicle directly ahead is at least "vehicle_ahead" with risk "medium", even if it is moving.
- A blocked ego lane, stopped vehicle, police/accident vehicle, debris, or vehicle across lane markings is "blocked_lane" with risk "high".
- If a lane change or overtake may be needed, mention that it is only safe after checking adjacent/oncoming traffic.
Focus on the ego lane, the next 30 meters, adjacent lanes, stopped vehicles, side collisions, pedestrians, traffic lights, and oncoming traffic."""


def extract_json(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in output: {text[:240]}")
    return json.loads(match.group(0))


def normalize_payload(payload, raw_text="", inference_seconds=0.0):
    status = {
        "scene_type": str(payload.get("scene_type", "unknown"))[:64],
        "risk_level": str(payload.get("risk_level", "unknown"))[:32],
        "main_hazard": str(payload.get("main_hazard", "unknown"))[:120],
        "safe_hint": str(payload.get("safe_hint", "observe_only"))[:160],
        "reason": str(payload.get("reason", ""))[:360],
        "raw_text": str(raw_text)[:1000],
        "inference_seconds": round(float(inference_seconds), 3),
    }
    try:
        status["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except Exception:
        status["confidence"] = 0.0
    return status


def mock_reason(frame_path):
    with Image.open(frame_path) as image:
        width, height = image.size
    return normalize_payload(
        {
            "scene_type": "unknown",
            "risk_level": "unknown",
            "main_hazard": "mock observer",
            "safe_hint": "vlm_cot_mock_only",
            "reason": f"Mock CoT received frame ({width}x{height}) but no VLM inference is active.",
            "confidence": 0.0,
        },
        raw_text="mock",
    )


class Qwen2VLBackend:
    def __init__(self, model_id, device="auto", max_new_tokens=180, load_in_4bit=False):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        if self.device == "cpu" and device == "auto":
            raise RuntimeError("CUDA is not available; refusing to load a 7B VLM on CPU in auto mode")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model_kwargs = {"trust_remote_code": True}

        if load_in_4bit:
            
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model_kwargs["device_map"] = "auto"
            # Leave some headroom below the full 4GB for Windows/other apps
            model_kwargs["max_memory"] = {0: "3.2GiB", "cpu": "24GiB"}
        else:
            model_kwargs["torch_dtype"] = dtype
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        if not load_in_4bit and self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def infer(self, frame_path):
        image = Image.open(frame_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)
        started = time.time()
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        payload = extract_json(output_text)
        return normalize_payload(payload, raw_text=output_text, inference_seconds=time.time() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--mode", default="mock", choices=["mock", "qwen2_vl"])
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true",
                         help="Quantize to 4-bit - needed to have any chance of fitting on a 4GB card")
    args = parser.parse_args()

    frames_dir = pathlib.Path(args.frames_dir)
    results_path = pathlib.Path(args.results_path or (frames_dir / "results.jsonl"))

    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    backend = None
    if args.mode == "qwen2_vl":
        backend = Qwen2VLBackend(args.model, load_in_4bit=args.load_in_4bit)

    with open(results_path, "w") as out:
        for filename, attack_label in ground_truth.items():
            frame_path = frames_dir / filename
            if not frame_path.exists():
                print(f"SKIP (missing file): {filename}")
                continue

            if args.mode == "mock":
                result = mock_reason(frame_path)
            else:
                result = backend.infer(frame_path)

            row = {"frame": filename, "ground_truth_attack": attack_label, **result}
            out.write(json.dumps(row) + "\n")
            print(f"{filename} | attack={attack_label} | model_risk={result['risk_level']} | hazard={result['main_hazard']}")

    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()