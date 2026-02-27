# app.py
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "画像が届いていません！"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "ファイルが選択されていません！"}), 400

    try:
        # 1. 届いた画像データをOpenCVで扱える形式に読み込む
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 2. ✨ここがAIエンジニアの腕の見せ所！擬似クリーニング処理✨
        
        # ① 画像を「BGR（青緑赤）」から「HSV（色相・彩度・明度）」という形式に変換
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)

        # ✨ ここに追加：元の画像の明度（V）の平均値を計算してスコア化
        original_brightness = np.mean(v)
        # 明度の平均値(0〜255)を100点満点に換算（intで小数点切り捨て）
        score = int((original_brightness / 255.0) * 100)

        # ② 汚れ（黄ばみや茶色い泥）を目立たなくするため、彩度(S)を半分(0.5倍)に落とす
        s = cv2.multiply(s, 0.5)

        # ③ 黒ずみを飛ばして白く見せるため、明度(V)を全体的に底上げする（+40）
        v = cv2.add(v, 40)

        # 分解して処理したHSVを、もう一度合体させる
        merged_hsv = cv2.merge((h, s, v))

        # 普通のカラー画像(BGR)に戻す
        cleaned_img = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)

        # ④ 仕上げ！コントラストを少し上げて「パリッと」した新品感（白の際立ち）を出す
        # alphaがコントラスト（1.0以上で強くなる）、betaが追加の明るさ
        final_img = cv2.convertScaleAbs(cleaned_img, alpha=1.1, beta=10)

        # 3. 綺麗になった画像を、ブラウザに送れる形式（Base64の文字列）にエンコードする
        _, buffer = cv2.imencode('.jpg', final_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 4. JSONに画像データを詰めてフロントエンドに返す
        return jsonify({
            "status": "success",
            "message": "クリーニング処理大成功！✨",
            "image_data": f"data:image/jpeg;base64,{img_base64}", 
            "score": score  # ⬅️ これを追加！
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"画像処理エラー: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)