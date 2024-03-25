from fastapi import FastAPI, UploadFile, File
from load import predictText

app = FastAPI()

@app.get("/")
def upload_file():
    try:
        file_path = "uploaded_files/" + 'Example.txt'
        text = ''
        with open(file_path, 'r', encoding="utf-8") as file:
            text = file.read()  # Считываем текст из файла
        outEx = predictText(text)
        return {"message": outEx}
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
