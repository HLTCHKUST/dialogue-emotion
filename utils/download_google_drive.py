import requests

def download_file_from_google_drive(id, destination):

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for i, chunk in enumerate(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
            if i % 100 == 0:
                print("download {} MB".format(3.2 * (i / 100))) 

if __name__ == "__main__":
    file_id = '1K0RPGSlBHOng4NN4Jkju_OkYtrmqimLi'
    destination = './emo2vec.zip'
    download_file_from_google_drive(file_id, destination)
