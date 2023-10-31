from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma

# 형태소 분석기 객체 생성
okt = Okt()  # Twitter 형태소 분석기
komoran = Komoran()  # 띄어쓰기가 잘 되어있지 않아도 분석이 뛰어남
hannanum = Hannanum()  # 카이스트 형태소 분석기
kkma = Kkma()  # 서울대학교 형태소 분석기
# mecab = Mecab()

text = "필리핀 경찰이 인신매매로 대규모 인력을 모아 범죄에 동원한 것으로 의심되는 조직을 급습해 한국인 포함 약 600명을 구금했다. 일부의 몸에는 고문당한 흔적이 발견됐는데 이들은 자신이 강제 감금됐다고 주장하고 있어, 수사당국은 용의자와 피해자를 구분하는 면담을 진행 중이다."

print('형태소 분석')
print("Okt:", okt.morphs(text))
print("Komoran:", komoran.morphs(text))
print("Hannanum:", hannanum.morphs(text))
print("Kkma:", kkma.morphs(text))

print('\n명사 분석')
print("Okt:", okt.nouns(text))
print("Komoran:", komoran.nouns(text))
print("Hannanum:", hannanum.nouns(text))
print("Kkma:", kkma.nouns(text))
