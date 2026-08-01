import argparse
import base64
import json
import os


def color_for_score(score, max_score):
    if max_score <= 0:
        ratio = 0
    else:
        ratio = min(1.0, score / max_score)
    if ratio < 0.33:
        return "#2ecc71"  # green
    elif ratio < 0.66:
        return "#f1c40f"  # yellow
    else:
        return "#e74c3c"  # red


def image_to_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output", default="test_outputs/surprise_report.html")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    max_score = max((r["surprise_score"] for r in results.values()), default=1.0)

    cards = []
    for name, r in results.items():
        frame_path = os.path.join(args.frames_dir, r.get("frame", ""))
        img_b64 = image_to_base64(frame_path)
        color = color_for_score(r["surprise_score"], max_score)

        img_html = (
            f'<img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:8px;">'
            if img_b64 else '<div style="height:150px;background:#333;border-radius:8px;'
                             'display:flex;align-items:center;justify-content:center;color:#888;">no image</div>'
        )

        cards.append(f"""
        <div style="background:#1e1e1e;border-radius:12px;padding:16px;width:280px;">
            {img_html}
            <h3 style="color:white;margin:12px 0 4px;">{name}</h3>
            <p style="color:#aaa;margin:2px 0;">Attack: {r.get('attack')}</p>
            <p style="color:#aaa;margin:2px 0;">Mapped scenario: {r.get('mapped_scenario')}</p>
            <div style="background:{color};color:black;font-weight:bold;padding:8px;
                        border-radius:8px;margin-top:10px;text-align:center;">
                Surprise: {r['surprise_score']:.4f}
            </div>
            <p style="color:#888;margin-top:6px;font-size:0.85em;">Risk head: {r.get('risk_head', 0):.3f}</p>
        </div>
        """)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>Surprise Report</title></head>
    <body style="background:#121212;font-family:sans-serif;padding:30px;">
        <h1 style="color:white;">WorldModel Surprise Report</h1>
        <p style="color:#aaa;">How confused was the model - normal vs attack scenarios</p>
        <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:20px;">
            {''.join(cards)}
        </div>
    </body>
    </html>
    """

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved to {args.output} - open it in a browser to view.")


if __name__ == "__main__":
    main()