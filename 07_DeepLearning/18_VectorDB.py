from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

examples = [
    """
    나루토의 친구는 누구인가요?
    최종 정답은 사스케
    """,
    """
    마법소녀 마도카☆마기카에서 마도카가 바라는 것은 무엇인가요?
    최종 정답은 모든 마법소녀들을 구원하기
    """,
    """
    마호로매틱에서 마호로는 어떤 직업을 가지고 있나요?
    최종 정답은 메이드 로봇
    """,
    """
    이추하고도 아름다운 세계에서 히카리는 누구인가요?
    최종 정답은 여신
    """,
    """
    미도리의 나날의 주인공 미도리는 무슨 색의 헤어를 가지고 있나요?
    최종 정답은 초록색
    """
]

db = Chroma.from_texts(
    collection_name="sample",
    texts=examples,
    embedding=HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
)

while True:
    # 사용자로부터 질문을 입력받습니다.
    question = input("질문을 입력하세요 (종료하려면 '종료' 입력): ")

    # '종료'가 입력되면 반복문을 빠져나갑니다.
    if question.lower() == '종료':
        print("프로그램을 종료합니다.")
        break

    # 질문에 대한 답을 검색합니다.
    doc = db.similarity_search(question, k=1)

    # 검색된 결과를 출력합니다.
    if doc:
        print("답변:", doc[0].page_content)
    else:
        print("해당 질문에 대한 답변을 찾을 수 없습니다.")
