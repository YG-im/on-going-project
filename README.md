# on-going-project 1 : Academic Writing Assistance Program for non-native speakers

## Motivation
- 학술 작문은 일반 작문과는 다르게 용어와 해당 필드의 언어 습관을 적절하게 활용하는 것이 굉장히 중요하다. 
- 하지만 생각보다 많은 사람들이 이를 간과한다.
- 하지만 이 사실을 알더라도 그 분야에서 논문을 몇 편써보기 이전에는 자신이 연구한 필드의 사람들이 관습적으로 사용하는 용어나 표현을 알아내기가 매우 어렵다.
- 좋은 연구를 하더라도 설득력 있게 글을 쓰기 위해서는 해당 분야의 논문들을 여러편을 정독하고 나오는 표현과 용어를 정리하는 작업을 선행해야한다.
- 원어민이라면 빠르게 할 수 있지만 비원어민에게는 어떻게 보면 연구 결과물을 얻어내는 시간보다 더 많은 시간과 노력을 요하는 작업이다.
- 또한, 논문 라이팅을 하다가도 특정 상황에서 기존 논문들이 통상적으로 사용하는 표현을 찾기위한 작업에 생각보다 많은 시간이 할애된다.
- 이 때문에 논문 작업의 효율이 크게 떨어지며, 설령 재밌는 아이디어가 떠올라도 새로운 분야에 대한 도전이 많이 망설이게 된다.
- 이러한 모티베이션 하에서 비원어민들이 논문의 라이팅이라는 작업보다 연구 그 자체에 더 몰두할수 있도록 학술 작문을 수월하게 진행할 수 있도록 도와주는 어휘, 표현, 문구 추천 알고리즘을 개발하고자 한다.

## 최종 목표
- 학술 작문을 하려는 분야의 선행 논문들(실제 연구에서 많이 참고한 레퍼런스들)을 머신에 학습시킨다.
- 머신은 레퍼런스들를 통해 자주쓰이는 어휘, 표현, 그리고 그 어휘, 표현이 나오는 문장의 패턴을 파악하게 된다.
- 그리고 연구자가 학술 작문을 하는 동안 머신은 연구자에게 적재적소에 적절한 어휘나 표현을 추천해준다.
- 연구자는 이를 참고하여 수월하게 논문 라이팅 작업을 수행할 수 있게 된다.

## 예상 사용층
- 영어를 어느정도 하지만 학술 작문에 사용되는 어휘나 특정 분야에서 쓰이는 표현 파악에 어려움을 겪는 연구자.
  - 이 알고리즘은 기본적으로 영어 단어나 표현을 추천해주는 알고리즘이다. 완성형 문장으로 만들고 이를 적재적소에 활용하는 것은 연구자의 몫이다. 따라서 아예 영어를 못하는 연구자는 사용하기 어려울 수 있다. 
  - 또한, 학습에 필요한 레퍼런스를 골라야하기 때문에 적어도 좋은 논문인지 아닌지 판단할 수 있는 능력이 연구자에게 필요하다.
  - 학위과정 중인 학생이라면 도움 없이 직접 작성하는 것을 몇 번 연습하고 사용하는 것을 추천한다.

## 머신러닝 구현 전략
### 1. 데이터 수집
- 사용자가 원하는 논문을 직접 수집해서 머신에 입력하는 방식으로 크롤링은 도입하지 않을 예정이다.
- 웹에 업로드되어있는 논문들은 pdf 형식이다. 
- pdf to txt를 해주는 프로그램을 활용하여 이들을 txt파일로 만들어준다.

### 2. 데이터 전처리
- 사용자가 원하는 pdf파일들을 입력하여 프로그램을 써서 txt파일로 만들 경우 글씨가 깨지거나 수식들이 깨져서 입력된 부분들이 상당히 많을 것이다.
- 영어 단어가 아닌 부분을 파악하여 제거하는 알고리즘을 구현한다.
- 저자명, 소속기관, 레퍼런스를 잘라내는 알고리즘을 구현한다.
- 이 단계의 알고리즘을 거치면 타이틀, 초록, 본문만 남은 논문이 txt파일로 출력된다.
- 제일 어려울 것이라고 예상됨.

### 3. 머신러닝 모델링
- Long Short-Trem Memory(LSTM) 모델을 차용한다.
- 학습된 알고리즘을 자동으로 저장해서 파일로 형성하는 시스템을 구현한다.
- 입력값이 들어가면 그에 따른 출력값(바로 뒤따라올 단어를 포함해 몇 개 단어 더)이 나오는 방식으로 추천 시스템을 구현한다.
- 최선의 답이 아니더라도 입력값에 뒤따라올 확률이 높은 순으로 몇 개를 추천해주는 시스텝을 구현한다.

### 4. Latex과 연동
- 완성된 프로그램을 논문 작성 프로그램인 Latex과 연동한다.

## To do list
- 데이터 수집.
 - 오픈 소스로 있는 pdf to txt프로그램들의 성능을 파악하고 문자 복원률을 확인한다.
 - 복원률이 낮으면 개선방안을 고안한다.
 
# on-going-project 2 : House price valuation (Kaggle prediction competitions)
- 집값을 예측하는 Kaggle 대회 참가하여 진행 중인 프로젝트입니다.
