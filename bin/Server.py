from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/receive_price', methods=['POST'])
def receive_price():
    data = request.get_json()
    print(f"Received: {data}")
    return jsonify({"status": "received", "symbol": data.get("symbol")}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
