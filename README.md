# chatbot
chatbot은 ChatGPT api를 사용하여 사용자의 질문에 대한 답변을 생성하는 모듈입니다.
## 개요
1. backend 서버로부터 사용자가 입력한 질문인 question을 받아 프롬프트를 생성합니다.
2-1. 경기도 정책 txt파일 문서를 /data 폴더에 준비합니다.
2-2. 정책 txt파일 문서들을 로드한 후, OpenAI에서 제공하는 OpenAIEmbeddings()를 이용해 임베딩 벡터를 생성합니다. 

3. Langchain 라이브러리의 RetrievalQA 함수를 사용하여,
    정책 txt 파일의 임베딩 벡터를 기반으로(retriever) 주어진 프롬프트에 대한 답변을 생성하는 qa_chain을 생성합니다. 
<img width="694" alt="image" src="https://github.com/user-attachments/assets/62d0ae13-3de4-4fdf-936b-f27758bad16e">

# 서버 실행
```uvicorn main:app --reload ```

# API 엔드포인트
## /chat : 정책 도메인이 정해지지 않았을 때, 사용자가 질문을 한다면 모든 정책을 기반으로 답변을 생성합니다.
input = ```{"question": "사용자의 질문 입력"}```
output =  ```{"answer": "생성된 답변"}```
## /policychat : 사용자가 특정 정책을 선택한 후(도메인이 결정된 상태), 사용자가 질문을 한다면 프롬프트에 "{policy_name}에 관련된 정책입니다."라는 문장을 추가하여 gpt가 정책을 특정하여 답변을 생성할 수 있도록 합니다. 
input = ```{"policy_name": "정책 이름", "question": "사용자의 질문 입력"}```
output =  ```{"answer": "생성된 답변"}```

# 데이터 구조

## data/*.txt : 정책 txt문서들을 load 한 후, embedding 하여 임베딩벡터를 retriever 으로 변환하여 검색기로 사용

## 의존성

- [FastAPI](https://fastapi.tiangolo.com/): API를 구축하기 위한 웹 프레임워크
- [langchain]: version langchain==0.0.347
- [OpenAI] : version openai==0.28.1
- [pydantic](https://pydantic-docs.helpmanual.io/): Python 타입 힌트를 사용한 데이터 유효성 검사 

```bash
pip install fastapi
pip install langchain==0.0.347
pip install openai==0.28.1
pip install pydantic
