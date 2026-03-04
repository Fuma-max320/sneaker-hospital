import cv2
import numpy as np
import base64
import traceback
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/care')
def care():
    return render_template('care.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "画像が届いていません！"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "ファイルが選択されていません！"}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 👇追加：画像が壊れている・認識できない場合のブロック
        if img is None:
            return jsonify({"status": "error", "message": "画像を正しく読み込めませんでした。別の写真でお試しください！"}), 400

        # 👇追加：超高画質な画像によるサーバーダウンを防ぐため、横幅最大800pxに縮小
        max_width = 800
        if img.shape[1] > max_width:
            ratio = max_width / img.shape[1]
            img = cv2.resize(img, None, fx=ratio, fy=ratio)

        # 画像をHSVに変換
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)

        # スコア計算
        original_brightness = np.mean(v)
        score = int((original_brightness / 255.0) * 100)

        # 彩度(S)を半分に（より確実な関数に変更）
        s = cv2.convertScaleAbs(s, alpha=0.5, beta=0)

        # 明度(V)の底上げ
        v = cv2.add(v, 40)

        # 合体させてBGRに戻す
        merged_hsv = cv2.merge((h, s, v))
        cleaned_img = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)

        # コントラスト調整
        final_img = cv2.convertScaleAbs(cleaned_img, alpha=1.1, beta=10)

        # Base64エンコード
        _, buffer = cv2.imencode('.jpg', final_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "status": "success",
            "message": "クリーニング処理大成功！✨",
            "image_data": f"data:image/jpeg;base64,{img_base64}",
            "score": score
        })

    except Exception as e:
        print("🚨🚨🚨 [画像処理エラー発生] 🚨🚨🚨")
        print(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": "AIがスニーカーをうまく認識できませんでした💦 もう少し明るい場所で、靴全体が写るように撮り直してみてください🙏"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)