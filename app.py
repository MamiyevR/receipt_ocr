from flask import Flask, request, jsonify
from utils import *
import logging

app = Flask(__name__)
app.config['TIMEOUT'] = 180
logger = logging.getLogger(__name__)


@app.route("/", methods=["POST"])
async def upload_image():
    try:
        data = request.get_json()

        # validate input
        validate_input_data(data)

        category_list = serialize_categories(data['categories'])

        # get ocr results
        predictions = get_ocr_predictions(data)

        # prepare gpt prompt
        payload, api_key = generate_prompt(predictions, data, category_list)

        # calling gpt
        gpt_result = await call_gpt(payload, api_key)

        # format the result
        result = format_result(gpt_result, data, category_list)

        # categorize
        # result = categorize_items(result, data["categories"])

        return jsonify(result)

    except InvalidInputException as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        logger.exception("An unexpected error occurred.")
        return jsonify({"error": "An unexpected error occurred."}), 500


if __name__ == "__main__":
    # os.environ["OPENAI_API_KEY"] = ""
    # os.environ["OPENAI_GPT4_API_KEY"] = ""
    # os.environ["BAIDU_API_KEY"] = ""
    # os.environ["BAIDU_SECRET_KEY"] = ""
    # os.environ["HF_API_KEY"] = ""
    # os.environ["VISION_API"] = ""

    app.run(debug=True, port=8000)
