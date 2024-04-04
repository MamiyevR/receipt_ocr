import json
import requests
import base64
import urllib
import aiohttp
import threading
import io
import PyPDF2
import os
from jellyfish import levenshtein_distance
from concurrent import futures

executor = futures.ThreadPoolExecutor(max_workers=2)


class InvalidInputException(Exception):
    pass


class APICallException(Exception):
    pass


def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded
    :return: base64编码信息
    """

    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)

    return content


def validate_input_data(data):
    """
    check the input data and run validation
    raises exception if data is not correct
    """

    required_keys = ["file_encoding", "ocr_model", "mode", "type", "categories", "company_name"]

    for key in required_keys:
        if not data.get(key):
            raise InvalidInputException(f"Missing {key} field in input data.")

    ocr_models_list = ["baidu", "vision"]
    if data["ocr_model"] not in ocr_models_list:
        raise InvalidInputException(f"Invalid ocr_model name, 'ocr_model' should be from {ocr_models_list}")

    gpt_models_list = ["gpt4", "gpt3-5"]
    if data["mode"] not in gpt_models_list:
        raise InvalidInputException(f"Invalid mode name, 'mode' should be from {gpt_models_list}")

    types_list = ["image", "pdf_file"]
    if data["type"] not in types_list:
        raise InvalidInputException(f"Invalid file type, 'type' should be from {types_list}")

    if len(data["categories"]) < 2:
        raise InvalidInputException("Too few categories")


def serialize_categories(categories):
    numbers_of_classes = {}
    for cat in categories:
        split = cat.split("::")
        cat_name = split[0]
        cat_num = split[1] if len(split) > 1 else ""

        numbers_of_classes[cat_name] = cat_num

    return numbers_of_classes


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": os.environ["BAIDU_API_KEY"],
        "client_secret": os.environ["BAIDU_SECRET_KEY"],
    }
    return str(requests.post(url, params=params).json().get("access_token"))


def baidu_call(url, headers, data):
    try:
        response = requests.request("POST", url, headers=headers, data=data)
        response.raise_for_status()
        # result = response.text
        return response.json()

    except requests.exceptions.RequestException as e:
        print("Failed to process image with Baidu OCR API:", str(e))
        return None

    except Exception as e:
        print("An error occurred:", str(e))
        return None


def baidu_async_calls(payloads, url, headers=None):
    results = []
    threads = []

    def call_baidu(payload):
        result = baidu_call(url, headers, payload)
        results.append(result)

    for payload in payloads:
        thread = threading.Thread(target=call_baidu, args=(payload,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results


def get_baidu_prediction(data):
    access_token = get_access_token()

    url = (
            "https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting?access_token="
            + access_token
    )

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }

    page_num = 1
    if data['type'] == "pdf_file" and 'page_num' in data:
        total_count = get_pdf_page_count(data['file_encoding'])
        page_num = min(int(data['page_num']), total_count, 2)

    payloads = []
    for i in range(1, page_num + 1):
        payload = f"{data['type']}={data['file_encoding']}&pdf_file_num={i}&" \
                  f"recognize_granularity=small&detect_direction=false&probability=false"
        payloads.append(payload)

    with executor:
        results = baidu_async_calls(payloads, url, headers)

    return results


def get_vision_prediction(data):
    try:
        if data['type'] == "pdf_file" and 'page_num' in data:
            vision_api_url = f"https://vision.googleapis.com/v1/files:annotate?key={os.environ['VISION_API']}"
            total_count = get_pdf_page_count(data['file_encoding'])
            page_num = min(int(data['page_num']), total_count, 5)

            body = {
                "requests": [{
                    "inputConfig": {
                        "content": urllib.parse.unquote_plus(data['file_encoding']),
                        "mimeType": "application/pdf"
                    },
                    "features": [{
                        "type": "DOCUMENT_TEXT_DETECTION"
                    }],
                    "pages": list(range(1, page_num+1))
                }]
            }

        else:
            vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={os.environ['VISION_API']}"

            body = {
                "requests": [{
                    "image": {
                        "content": urllib.parse.unquote_plus(data['file_encoding'])
                    },
                    "features": [{
                        "type": "DOCUMENT_TEXT_DETECTION"
                    }]
                }]
            }

        response = requests.post(vision_api_url, json=body)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print("Failed to process image with Google Vision API:", str(e))
        return None

    except Exception as e:
        print("An error occurred:", str(e))
        return None


def get_ocr_predictions(data):
    if data['ocr_model'] == "baidu":
        return get_baidu_prediction(data)
    else:
        return get_vision_prediction(data)


def generate_prompt(predictions, data, category_list):
    with open("data/_promptTemplateHeader.txt", "r") as file:
        prompt_template = file.read()

    words_string = ""

    if data['ocr_model'] == "baidu":
        for page in predictions:
            for word in page["words_result"]:
                words_string += word["words"] + "\n"

        prompt_template = prompt_template + "\n" + words_string

    elif data['type'] == "pdf_file":
        for page in predictions['responses'][0]['responses']:
            prompt_template = prompt_template + "\n" + page['fullTextAnnotation']['text']

    else:
        prompt_template = prompt_template + "\n" + predictions['responses'][0]['fullTextAnnotation']['text']

    return prepare_payload(prompt_template, data['mode'], category_list)


def prepare_payload(prompt_text, mode, category_list):
    with open(f"data/_schema.json") as json_file:
        json_schema = json.load(json_file)

    json_schema['properties']['file_items']['items']['properties']['category']['enum'] = list(category_list.keys())

    if mode == "gpt3-5":
        model_name = "gpt-3.5-turbo-1106"
        api_key = os.environ["OPENAI_API_KEY"]

    else:
        model_name = "gpt-4-1106-preview"
        api_key = os.environ["OPENAI_GPT4_API_KEY"]

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful accountant assistant"
            },
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "functions": [
            {
                "name": "set_json",
                "parameters": json_schema,
            },
        ],
        "function_call": {
            "name": "set_json"
        },
        "temperature": 0,
        "response_format": {
            "type": "json_object"
        },
        "seed": 99
    }

    return payload, api_key


async def call_gpt(payload, api_key):
    request_url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url=request_url, headers=headers, json=payload, ssl=False) as response:
                return await response.json()

        except aiohttp.ContentTypeError as e:
            response_text = await response.text()
            print(f"Received unexpected content type: {response.content_type}")
            print(f"Response content: {response_text}")
            return None

        except Exception as e:
            print(f"An error occurred: {e}")
            return None


def trim_result(prompt_result):
    result = prompt_result['choices'][0]['message']['content']

    start_index = result.find("{")
    end_index = result.rfind("}")

    start_index = 0 if start_index == -1 else start_index
    end_index = len(result) - 1 if end_index == -1 else end_index

    result = result[start_index:end_index + 1]

    return result


def format_result(gpt_result, data, category_list):
    gpt_result = gpt_result["choices"][0]["message"]["function_call"]["arguments"]

    company_name = data['company_name']
    gpt_result = json.loads(gpt_result)
    f_type = gpt_result['file_type'].lower()
    gpt_result['my_merchant_name'] = company_name

    if 'created_by_merchant_name' not in gpt_result:
        gpt_result['created_by_merchant_name'] = ''
    if 'bill_to_merchant_name' not in gpt_result:
        gpt_result['bill_to_merchant_name'] = ''
    if 'merchant_address' not in gpt_result:
        gpt_result['merchant_address'] = ''
    if 'note' not in gpt_result:
        gpt_result['note'] = ''
    if 'file_items' not in gpt_result:
        gpt_result['file_items'] = []
    if 'file_number' not in gpt_result:
        gpt_result['file_number'] = ''
    if 'payment_method' not in gpt_result:
        gpt_result['payment_method'] = 'Cash'

    if f_type == 'invoice':
        # set merchant_name
        if lev_sim(company_name, gpt_result['bill_to_merchant_name']) > 0.7:
            gpt_result['merchant_name'] = gpt_result['created_by_merchant_name']
        else:
            gpt_result['merchant_name'] = gpt_result['bill_to_merchant_name']

        # set file_category
        if lev_sim(company_name, gpt_result['created_by_merchant_name']) > 0.7:
            gpt_result['file_category'] = 'in_flow'
        else:
            gpt_result['file_category'] = 'out_flow'

    elif f_type == 'receipt':
        # set merchant_name
        if lev_sim(company_name, gpt_result['created_by_merchant_name']) > 0.7:
            gpt_result['merchant_name'] = gpt_result['bill_to_merchant_name']
        else:
            gpt_result['merchant_name'] = gpt_result['created_by_merchant_name']

        # set file_category
        if lev_sim(company_name, gpt_result['created_by_merchant_name']) > 0.7:
            gpt_result['file_category'] = 'in_flow'
        else:
            gpt_result['file_category'] = 'out_flow'

    elif f_type == 'payroll':
        # set merchant_name
        gpt_result['merchant_name'] = company_name
        gpt_result['created_by_merchant_name'] = company_name

        # set file_category
        gpt_result['file_category'] = 'out_flow'

    for item in gpt_result['file_items']:
        if 'quantity' not in item:
            item['quantity'] = 1

        if item['category'] in category_list:
            item['category'] = item['category'] + "::" + category_list[item['category']]

    return gpt_result


def lev_sim(str1, str2):
    string1 = str1.lower()
    string2 = str2.lower()
    dist = levenshtein_distance(string1, string2)
    similarity = 1 - (dist / max(len(string1), len(string2)))

    return similarity


# divide long list to 10 element subsets
def divide_list(category_keys):
    subset_size = 10
    subsets = []
    remaining_elements = []

    for i, element in enumerate(category_keys):
        if i % subset_size == 0:
            subsets.append([])

        if len(subsets) <= i // subset_size:
            remaining_elements.append(element)
        else:
            subsets[i // subset_size].append(element)

    if len(remaining_elements) > 0:
        subsets.append(remaining_elements)

    return subsets


def call_classifier(api_url, headers, item, reduced_cat):
    # hugging face inference api only handle 10 classes at a time
    # so, we divide overall list to 10 element sublist and call api on sublist
    divided_list = divide_list(reduced_cat)

    label_prob = {}
    for sub_list in divided_list:
        payload = {
            "inputs": item['name'],
            "parameters": {
                "candidate_labels": sub_list,
                "multi_label": True
            }
        }

        response = requests.post(api_url, headers=headers, json=payload)
        # print(response.json())

        i = 0
        while response.status_code != 200 and i < 3:
            response = requests.post(api_url, headers=headers, json=payload)
            i += 1

        # print(response.json())
        if response.status_code == 200:
            probs = response.json()

            for label, prob in zip(probs['labels'], probs['scores']):
                label_prob[label] = prob

        else:
            for label in reduced_cat:
                label_prob[label] = 0

            break

    return label_prob


def validate(result):
    # check if all values of dictionary 0
    for key in result.keys():
        if result[key] > 0:
            return True

    return False


def remove_failed(results):
    # remove results with all 0 probability
    # i.e. failed from categorization api
    new_results = []
    for result in results:
        if validate(result):
            new_results.append(result)

    return new_results


def categorize_items(result, categories):
    api_url = "https://api-inference.huggingface.co/models/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    headers = {"Authorization": "Bearer " + os.environ["HF_API_KEY"]}

    with open(f"data/categories_v2.json") as json_file:
        categories_list = json.load(json_file)

    f_type = result['file_type']
    reduced_cat = {}
    numbers_of_classes = {}

    for cat in categories:
        splitted = cat.split("::")
        cat_name = splitted[0]
        cat_num = splitted[1] if len(splitted) > 1 else ""

        numbers_of_classes[cat_name] = cat_num
        if cat_name in categories_list:
            if f_type in categories_list[cat_name]["types"]:
                for subcat in categories_list[cat_name]["subcategories"]:
                    if subcat in reduced_cat:
                        reduced_cat[subcat].append(cat_name)
                    else:
                        reduced_cat[subcat] = [cat_name]
        else:
            reduced_cat[cat_name] = [cat_name]

    category_keys = list(reduced_cat.keys())
    classification_results = []
    threads = []

    for item in result['file_items']:
        thread = threading.Thread(target=lambda: classification_results.append(
            call_classifier(api_url, headers, item, category_keys)))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    total_probs = {}

    classification_results = remove_failed(classification_results)
    # print(classification_results)
    item_num = len(classification_results)

    if item_num > 2:
        for class_name in reduced_cat:
            total_prob = 0
            for item in classification_results:
                total_prob += item[class_name]

            total_probs[class_name] = total_prob / item_num

        max_pair = max(total_probs.items(), key=lambda x: x[1])
        new_category = reduced_cat[max_pair[0]][0]

        for item in result['file_items']:
            if numbers_of_classes[new_category]:
                item['category'] = "".join([new_category, "::", numbers_of_classes[new_category]])
            else:
                item['category'] = new_category

    return result


def get_pdf_page_count(url_encoding):
    base64_encoded_pdf = urllib.parse.unquote_plus(url_encoding)
    pdf_data = base64.b64decode(base64_encoded_pdf)
    pdf_file = io.BytesIO(pdf_data)

    pdf_reader = PyPDF2.PdfReader(pdf_file)
    page_count = len(pdf_reader.pages)

    return page_count


if __name__ == "__main__":
    file_url = get_file_content_as_base64("data/doc.jpg", True)

    with open("output.txt", "w") as txt_file:
        txt_file.write(file_url)
