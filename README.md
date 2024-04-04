# receipt-ocr

The body is in the following format:
```
{
    "file_encoding": string,  //base64, url_encoded file
    "ocr_model": string       //baidu, vision
    "mode": string,           //gpt4, gpt3-5
    "type": string,           //type of file (image, pdf_file)
    "page_num": int,          //number of pages to process
                              //ignored if type is image
    "categories": list,       //list of categories
    "company_name": string    //company name
}
```
To run locally, need to include secret keys for GPT, BAIDU, Hugging Face
