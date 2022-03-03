## NewsTrust

### [KPFBERT / Newstrust](https://github.com/KPFBERT/Newstrust)

### Test Summary
- 정형화된 기사 데이터를 바탕으로 기사의 타이틀과 컨텐츠를 분석
- 분석 결과를 토대로 스코어를 출력
- 소스 코드 다운로드 및 실행
- 동작 환경 구성 및 패키지 의존성 풀기
- 실행 결과 확인

### TODO
- 소스 코드 분석

### Issues
- 전체적으로 테스트 실행을 위한 설명이 부족하며 패키지 의존성 해결을 위한 추가 정리 필요
- 테스트 구동 시 필요한, 파이썬 한글 맞춤법 검사 라이브러리는 리눅스에서만 사용 가능
```
https://github.com/ssut/py-hanspell
```
- 구동을 위하여 git-lfs 패키지 설치 필요: 버전 2.9에서는 동작하지 않음
```bash
wget https://packagecloud.io/github/git-lfs/packages/debian/buster/git-lfs_2.10.0_amd64.deb/download
```
---

### Environment
```
- Oracle VM VirtualBox / Ubuntu 20.04.2 LTS
- Python 3.7.10
```

### Sample Code
- 기 작성된 코드 구동

### Test News Data
- 정형화된 뉴스 데이터를 기반으로 분석을 진행
- 코드에서 제공하는 샘플 데이터
```json
[
  {
    "news_id": "07100501.20210601105013001",
    "title": "산단공, 코로나19 예방 백신 특별휴가제도 도입",
    "content": "한국산업단지공단은 노동조합과 코로나19 백신 예방 접종 촉진을 위한 특별휴가제도를 도입했다고 1일 밝혔다.\n\n산단공 노사가 합의한 이번 제도는 접종 당일과 익일 특별휴가를 부여하고, 이상 반응이 있으면 추가로 하루 특별휴가를 사용할 수 있는 형태다.\n\n김정환 산단공 이사장은 “백신 예방 접종으로 집단 면역을 형성하는 것은 우리 국민 일상 회복을 위해 반드시 실행해야 하는 의무”라고 강조했다.\n\n윤희석기자 pioneer@etnews.com",
    "published_at": "2021-06-01T00:00:00.000+09:00",
    "enveloped_at": "2021-06-01T10:50:13.000+09:00",
    "dateline": "2021-06-02T10:30:12.000+09:00",
    "provider": "전자신문",
    "category": [
      "사회>의료_건강",
      "정치>정치일반",
      "지역>경기"
    ],
    "category_incident": [],
    "byline": "윤희석",
    "images": "/07100501/2021/06/01/07100501.20210601105013001.01.jpg",
    "provider_subject": [
      {
        "subject_info1": "",
        "subject_info2": "",
        "subject_info3": "",
        "subject_info4": "",
        "subject_info": "산업_과학_정책"
      }
    ],
    "provider_news_id": "20210601000087",
    "publisher_code": "",
    "fix_category": "연예"
  },
  {
    "news_id": "07100501.20210601144623001",
    "title": "편의점 '세 번' 갈 때…쿠팡・배민 '네 번' 시켰다",
    "content": "코로나19 발생 이후 쿠팡과 배달의민족의 구매 빈도가 편의점을 앞지른 것으로 나타났다. 외부 활동 위축으로 집에서 온라인쇼핑, 배달 음식 등 비대면 소비가 증가한 데 따른 것으로 분석된다. 비대면 채널에서 더 자주 구매하는 고객이 늘면서 온라인 플랫폼의 장기적 성장 기반 마련에 긍정적으로 작용할 것이 전망된다.\n\n1일 시장조사업체 오픈서베이가 패널 1만명을 대상으로 지난해 2월부터 올해 1월까지 누적 신용·체크카드 결제 내역 데이터를 분석한 결과 쿠팡의 월 결제건수는 8800건에서 1만500건으로 늘었고, 배민 역시 4700건에서 7900건으로 증가했다. 조사는 카테고리·브랜드별 결제금액과 객단가, 결제건수, 월별 결제자수 등의 지표를 분석해서 이뤄졌다. 쿠팡과 배민은 코로나19 유행 기간에 카드결제 내역이 늘어난 상위 2개 브랜드로 집계됐다. 특히 1인당 월평균 결제 빈도에서 소액 결제 비중이 높은 편의점마저 넘어선 것으로 조사됐다.\n\n쿠팡은 전체 고객의 25%가 월평균 4.2회 결제했고, 배민에서는 전체의 23%가 월평균 3.4회 결제한 것으로 나타났다. 반면 같은 기간 편의점 브랜드 GS25와 CU의 월평균 결제 빈도는 각각 3.2회, 3.0회로 집계됐다. 코로나19 발생 이후 소비자들이 쿠팡과 배민을 편의점보다 더 자주 이용했음을 보여 준다.\n\n편의점의 경우 지난해 객단가는 높아졌지만 집합금지 등 외부 활동이 제한된 영향으로 구매 빈도는 낮아졌다. 산업통상자원부에 따르면 편의점 구매건수는 지난해 3월 15.3% 역신장 이후 올해 2월까지 11개월 연속 감소했다. 번화가·대학가 근처 편의점 방문이 끊기고 야간 유동인구가 줄면서 연쇄적으로 영향을 받았다.\n\n반면 온라인 플랫폼에 대한 결제 건수 증가는 매출 성장으로 이어졌다. 쿠팡의 지난해 매출은 13조9235억원으로 전년(7조1530억원) 대비 94% 증가했다. 거래액도 약 21조원의 66% 늘어난 것으로 추산된다. 배민을 운영하는 우아한형제들 역시 지난해 매출이 1조995억원으로 전년 대비 94.4% 늘었다.\n\n쿠팡은 특히 식료품 판매가 늘었다. 신선식품 등 먹거리를 오프라인 대신 온라인으로 구매하는 수요가 급격하게 증가하면서 쿠팡이 수혜를 누린 것으로 분석된다. 실제로 지난해 온라인 식품시장 거래액은 43조4000억원으로 전년보다 62.4% 성장했다.\n\n배민도 음식 배달시장 성장과 맞물렸다. 지난해 음식서비스의 온라인 거래액은 17조4000억원으로 전년 대비 78.6% 늘었다. 올 1분기에도 온라인 음식서비스 거래액은 지난해 같은 기간 대비 71.9% 늘며 1분기 전체 온라인쇼핑 거래액 신장률(21.3%)을 웃돌았다. 업계 관계자는 “코로나19 유행 기간에 높은 성장세를 보인 서비스 대부분이 한 번에 지출하는 비용보다 결제건수가 꾸준히 증가했다”면서 “1인당 결제 빈도가 늘었다는 것은 더 장기적인 성과를 기대할 수 있게 한다”고 말했다.\n\n박준호기자 junho@etnews.com",
    "published_at": "2021-06-01T00:00:00.000+09:00",
    "enveloped_at": "2021-06-01T14:46:23.000+09:00",
    "dateline": "2021-06-02T10:56:05.000+09:00",
    "provider": "전자신문",
    "category": [
      "경제>유통",
      "IT_과학>모바일",
      "경제>자동차"
    ],
    "category_incident": [],
    "byline": "박준호",
    "images": "/07100501/2021/06/01/07100501.20210601144623001.01.jpg",
    "provider_subject": [
      {
        "subject_info1": "",
        "subject_info2": "",
        "subject_info3": "",
        "subject_info4": "",
        "subject_info": "전자_자동차_유통"
      }
    ],
    "provider_news_id": "20210601000212",
    "publisher_code": "",
    "fix_category": "경제"
  },
  {
    "news_id": "07100501.20210601120912001",
    "title": "다온플래닛, 인프라 모니터링 서비스 지원",
    "content": "다온플래닛(대표 임종화)이 인프라 모니터링 서비스 'DIMS(Daonplanet Infra Monitoring Service)'를 출시하고 기업 인프라 시장 공략에 나섰다. 회사는 자체 개발한 모니터링 프로그램과 협력사 프로그램을 결합해 중소기업을 대상으로 솔루션 공급에 나선다.\n\nDIMS는 간이 네트워크 관리 프로토콜(SNMP)과 VPN 터널링을 기반으로 하는 인프라 모니터링 서비스다. VPN 터널링은 VPN으로 연결한 두 지점 간에 가상 통로를 생성하는 것이다. 이 과정에서 사설망과 같은 보안 기능을 제공한다.\n\n서비스는 기업 내부에서 쓰이는 △방화벽 △스위치 △서버 △PC △무선컨트롤러 △모바일 AP를 대상으로 로그 데이터와 여러 정보를 취합한 네트워크 보안 리포트를 고객사에 제공한다. 인프라 장애가 발생할 경우 자동으로 고객사 메일과 소셜미디어를 통해 알림 정보를 제공한다. 고객사에 관련 유지 보수 서비스도 제공한다. 솔루션 적용에 별도 통합보안관리(ESM)와 네트워크 관리 시스템(NMS) 구축이 필요없다.\n\n임종화 대표는 “기업의 올바른 인프라 네트워크 구성을 지원할 것”이라며 “중소기업 기업 환경에 최적화한 네트워크 보안 환경을 제공할 계획”이라고 말했다. 이어 “가용성과 안정성, 효율성을 잡을 수 있는 최적 솔루션이라고 자부한다”고 덧붙였다.\n\n임중권기자 lim9181@etnews.com",
    "published_at": "2021-06-01T00:00:00.000+09:00",
    "enveloped_at": "2021-06-01T12:09:12.000+09:00",
    "dateline": "2021-06-01T15:28:20.000+09:00",
    "provider": "전자신문",
    "category": [
      "IT_과학>보안",
      "IT_과학>모바일",
      "IT_과학>콘텐츠"
    ],
    "category_incident": [],
    "byline": "임중권",
    "images": "/07100501/2021/06/01/07100501.20210601120912001.01.jpg",
    "provider_subject": [
      {
        "subject_info1": "",
        "subject_info2": "",
        "subject_info3": "",
        "subject_info4": "",
        "subject_info": "SW_게임_성장기업"
      }
    ],
    "provider_news_id": "20210601000119",
    "publisher_code": "",
    "fix_category": "IT 과학"
  },
    {
    "news_id": "07100501.20210601120912001",
    "title": "‘메탄올 뿌려서 소독?’…또 다른 재난 ‘인포데믹’ 피하려면?",
    "content": "#사례1.⏎\n메탄올 뿌려 소독하려다.⏎\n..⏎\n남양주에 사는 40대 여성 A씨는 지난 7일 메탄올을 섞은 물을 집안에 뿌렸습니다.⏎\n분무기로 가구와 이불 곳곳에 이 액체를 분사했습니다.⏎\n메탄올을 섞은 물을 뿌리면 코로나19 바이러스가 없어질 것이라는 생각에서였습니다.⏎\n그러나 결과는 위험천만했습니다.⏎\nA씨는 곧이어 복통, 구토, 어지럼증을 보였습니다.⏎\nA의 자녀 2명도 같은 증세를 보였습니다.⏎\n메탄올 중독 증상이었습니다.⏎\nA씨 등은 결국 병원 신세를 져야 했습니다.⏎\n안전보건공단, \"메탄올 소독에 사용하면 위험\"⏎\nA씨가 소독에 메탄올을 사용한 사실을 확인한 안전보건공단은 \"코로나19 방역을 위해 메탄올을 사용해서는 안된다\"며 \"메탄올은 인화성이 강한 무색 액체로 반복 노출되면 중추신경계와 시신경에 손상을 유발하는 독성 물질\"이라고 밝혔습니다.⏎\n코로나19 소독은 커녕오히려 건강을 해칠 수 있다는 이야기입니다.⏎\n당연히 방역을 위해 사용하면 안 됩니다.⏎\n#사례2.⏎\n소금물로 구강 소독?⏎\n분무기를 거의 입 안에 넣을 듯이 깊게 들이대고 무엇인가를 분사하는 이 사진.⏎\n이번에는 소금물입니다.⏎\n소금물을 입안에 뿌리면 효과가 있을 것이라는 생각에 예배를 보는 신도들 입안에 뿌린 것입니다.⏎\n입안 뿐만 아니라 손에도 뿌렸습니다.⏎\n바로 70명이 넘는 코로나19 확진자가 발생한 '은혜의 강 교회' 예배 당시의 모습입니다.⏎\n\"사실상 직접 접촉과 다름없어\"⏎\n이 사진을 공개한 이희영 경기도 코로나19 긴급대책단 공동단장은 \"예배 참석자 중에 확진자가 있었고 분무기를 소독하지 않고 계속 뿌렸기 때문에 사실상의 직접접촉이나 다름이 없다고 추정한다\"고 밝혔습니다.⏎\n그러면서 \"잘못된 정보로 인한 인포데믹(Infomemic, 정보감염증)\"이라고 설명했습니다.⏎\n바이러스 감염 만큼 위험한 정보감염⏎\n생명을 지키려는 행동이 잘못된 정보, 허위 정보에 기반을 두게 되면 오히려 생명을 위협합니다.⏎\n인포데믹(infodemic).⏎\n정보(infomation)과 전염병(endemic)의 합성어입니다.⏎\n잘못된 정보도 전염병처럼 급속도로 퍼져나가 혼란과 공공안전을 위협한다는 의미입니다.⏎\n감염병 대유행과 같은 급박한 상황에서 정보감염은 심각한 결과를 초래할 수 있습니다.⏎\n해외에서도 정보감염에 따른 혼란이 이어지고 있습니다.⏎\n미국에서는 최근 메사추세츠 주지사가 자택대피명령(shelter in place)을 내린다는 소문이 소셜 미디어를 통해 퍼져나갔습니다.⏎\n혼란에 사로잡힌 사람들은 상점의 파스타, 휴지, 식료품을 사재기했습니다.⏎\n급기야 찰리 베이커 주지사가 자택대피명령을 내릴 계획은 없다고 해명에 나서기도 했습니다.⏎\n그러면서 베이커 주지사는 \"코로나19에 대한 정보는 신뢰할 수 있는 기관에서 얻어야 한다.⏎\n친구의, 친구의, 친구의, 친구의 이웃(their \"friend's friend's friend's friend's neighbor)에서가 아니라.⏎\n\"라고 충고했습니다.⏎\nWHO의 EPI-WIN 인터넷 홈페이지⏎\n'인포데믹'에 맞서는 국제 기관들⏎\n이렇게 전세계로 퍼지는 인포데믹에 맞서 팩트체크 기관들도 연대했습니다.⏎\n국제 팩트체킹 네트워크(International Fact-Checking Network, IFCN)은 지난 1월 100곳이 넘는 세계의 팩트체크 기관과 연합해 코로나19와 관련된 허위, 가짜정보에 대한 팩트체크를 시작했습니다.⏎\n코로나19 뿐만 아니라 인포데믹에 맞서는 전세계적 움직임도 진행되고 있는 겁니다.⏎\n세계보건기구(WHO)도 EPI-WIN이라는 정보 플랫폼을 온라인상에서 운영하고 있습니다.⏎\n감염병 대유행 등 위험 상황에서는 정보에 대한 수요가 폭증하고 허위 정보들이 이러한 틈을 파고들어 확산됩니다.⏎\n세계보건기구는 허위 정보보다는 신뢰할 수 있는 정보들을 증폭, 확산하겠다는 의도로 이 정보 플랫폼을 운영하고 있습니다.⏎\n기본적인 코로나19에 대한 정보와 함께 주기적인 정보 업데이트를 제공하고 있습니다.⏎\n[연관링크] WHO가 운영하는 정보플랫폼, EPI-WIN⏎\n인포데믹에 감염되지 않으려면?⏎\n정답은 나와 있습니다.⏎\n정보에 대한 철저한 출처 확인입니다.⏎\n인터넷에서 코로나19 관련 정보를 접하면 그 정보의 출처와 정보원이 어디인지 확인하는 것입니다.⏎\n출처가 불분명명하고 구전이나 소셜미디어를 통해 전파되는 정보는 허위정보일 위험성이 크다고 전문가들은 지적합니다.⏎\n정보의 출처를 확인하고 의심이 드는 정보는 확산시키지 말아야 합니다.⏎\n코로나19의 세계적 대유행은 인포데믹이라는 또 다른 재난을 낳았지만 그것을 극복할 수 있는 것은 우리의 냉철한 이성입니다.⏎\n▶ ‘ 코로나19 확산 우려’ 최신 기사 보기 http://news.kbs.co.kr/news/list.do?icd=19588⏎\n▶ ‘코로나19 팩트체크’ 제대로 알아야 이긴다 바로가기 http://news.kbs.co.kr/issue/IssueView.do?icd=19589⏎\n▶ 우리동네에서 무슨일이?⏎\nKBS지역뉴스 바로가기 http://news.kbs.co.kr/local/main.do⏎\n박희봉 기자 (thankyou@kbs.co.kr)",
    "published_at": "2021-06-01T00:00:00.000+09:00",
    "enveloped_at": "2021-06-01T12:09:12.000+09:00",
    "dateline": "2021-06-01T15:28:20.000+09:00",
    "provider": "KBS",
    "category": [
      "사회_생활"
    ],
    "category_incident": [],
    "byline": "박희봉",
    "images": "/07100501/2021/06/01/07100501.20210601120912001.01.jpg",
    "provider_subject": [
      {
        "subject_info1": "",
        "subject_info2": "",
        "subject_info3": "",
        "subject_info4": "",
        "subject_info": ""
      }
    ],
    "provider_news_id": "20210601000119",
    "publisher_code": "",
    "fix_category": "사회"
  }

]
```

### Result
- 테스트 실행 결과
```
----------print Analysis Title----------
TitleText :  산단공, 코로나19 예방 백신 특별휴가제도 도입
TitleMecabTag :  [('산', 'NNG'), ('단공', 'NNG'), (',', 'SC'), ('코로나', 'NNP'), ('19', 'SN'), ('예방', 'NNG'), ('백신', 'NNG'), ('특별', 'NNG'), ('휴가제', 'NNG'), ('도', 'JX'), ('도입', 'NNG')]
TitleLen :  26
QuestionMarkCount :  0
ExclamationMarkCount :  0
TitleAdverbsCount :  0
TitleDoubleQuotationsMarksNum :  0
----------End Analysis Title----------
5
한국산업단지공단은 노동조합과 코로나19 백신 예방 접종 촉진을 위한 특별휴가제도를 도입했다고 1일 밝혔다
산단공 노사가 합의한 이번 제도는 접종 당일과 익일 특별휴가를 부여하고, 이상 반응이 있으면 추가로 하루 특별휴가를 사용할 수 있는 형태다
김정환 산단공 이사장은 "백신 예방 접종으로 집단 면역을 형성하는 것은 우리 국민 일상 회복을 위해 반드시 실행해야 하는 의무"라고 강조했다
윤희석기자 pioneer@etnews.com
4
----------print Analysis Content----------
Provider :  전자신문
Category :  연예
ContentLen :  243
ContentNumericalCitationNum :  2
AverageSentenceLen :  60.25
AverageAdverbSentenceNum :  0.25
AverageQuotesNum :  1
AverageQuotesLen :  0.23045267489711935
QuotesBuffer :  "백신 예방 접종으로 집단 면역을 형성하는 것은 우리 국민 일상 회복을 위해 반드시 실행해야 하는 의무"
----------End Analysis Content----------
----------print Value----------
BylineScore :  1
ContentLenScore :  0
QuotesNumScore :  0.06666666666666667
TitleLenScore :  0
TitleQuestionMarkExclamationMarkScore :  0
ContentNumericalCitationNumScore :  0
AverageSentenceLenScore :  0
TitleAdverbsCountScore :  0
AverageAdverbSentenceNumScore :  0
AverageQuotesLenScore :  0
Journal :  {'scoreTotal': 34.18973333333333, 'readability': 0.0010666666666666667, 'transparency': 4.798, 'factuality': 4.7264, 'utility': 3.6274, 'fairness': 3.196, 'diversity': 1.1647333333333334, 'originality': 4.727399999999999, 'importance': 2.7283333333333335, 'depth': 4.7294, 'sensationalism': 4.491}
Vanilla :  {'scoreTotal': 10.6, 'readability': 1.0666666666666667, 'transparency': 1.0666666666666667, 'factuality': 1.0666666666666667, 'utility': 1.0666666666666667, 'fairness': 1.0666666666666667, 'diversity': 1.0666666666666667, 'originality': 1.0666666666666667, 'importance': 1.0666666666666667, 'depth': 1.0666666666666667, 'sensationalism': 1.0}
----------End Value----------
----------print Analysis Title----------
TitleText :  편의점 '세 번' 갈 때…쿠팡・배민 '네 번' 시켰다
TitleMecabTag :  [('편의점', 'NNG'), ("'", 'SY'), ('세', 'MM'), ('번', 'NNBC'), ("'", 'SY'), ('갈', 'VV+ETM'), ('때', 'NNG'), ('…', 'SE'), ('쿠팡', 'NNP'), ('・', 'SL'), ('배민', 'NNG'), ("'", 'SY'), ('네', 'MM'), ('번', 'NNBC'), ("'", 'SY'), ('시켰', 'VV+EP'), ('다', 'EC')]
TitleLen :  29
QuestionMarkCount :  0
ExclamationMarkCount :  0
TitleAdverbsCount :  0
TitleDoubleQuotationsMarksNum :  0
----------End Analysis Title----------
36
코로나19 발생 이후 쿠팡과 배달의민족의 구매 빈도가 편의점을 앞지른 것으로 나타났다
 외부 활동 위축으로 집에서 온라인쇼핑, 배달 음식 등 비대면 소비가 증가한 데 따른 것으로 분석된다
 비대면 채널에서 더 자주 구매하는 고객이 늘면서 온라인 플랫폼의 장기적 성장 기반 마련에 긍정적으로 작용할 것이 전망된다
1일 시장조사업체 오픈서베이가 패널 1만명을 대상으로 지난해 2월부터 올해 1월까지 누적 신용·체크카드 결제 내역 데이터를 분석한 결과 쿠팡의 월 결제건수는 8800건에서 1만500건으로 늘었고, 배민 역시 4700건에서 7900건으로 증가했다
 조사는 카테고리·브랜드별 결제금액과 객단가, 결제건수, 월별 결제자수 등의 지표를 분석해서 이뤄졌다
 쿠팡과 배민은 코로나19 유행 기간에 카드결제 내역이 늘어난 상위 2개 브랜드로 집계됐다
 특히 1인당 월평균 결제 빈도에서 소액 결제 비중이 높은 편의점마저 넘어선 것으로 조사됐다
쿠팡은 전체 고객의 25%가 월평균 4
2회 결제했고, 배민에서는 전체의 23%가 월평균 3
4회 결제한 것으로 나타났다
 반면 같은 기간 편의점 브랜드 GS25와 CU의 월평균 결제 빈도는 각각 3
2회, 3
0회로 집계됐다
 코로나19 발생 이후 소비자들이 쿠팡과 배민을 편의점보다 더 자주 이용했음을 보여 준다
편의점의 경우 지난해 객단가는 높아졌지만 집합금지 등 외부 활동이 제한된 영향으로 구매 빈도는 낮아졌다
 산업통상자원부에 따르면 편의점 구매건수는 지난해 3월 15
3% 역신장 이후 올해 2월까지 11개월 연속 감소했다
 번화가·대학가 근처 편의점 방문이 끊기고 야간 유동인구가 줄면서 연쇄적으로 영향을 받았다
반면 온라인 플랫폼에 대한 결제 건수 증가는 매출 성장으로 이어졌다
 쿠팡의 지난해 매출은 13조9235억원으로 전년(7조1530억원) 대비 94% 증가했다
 거래액도 약 21조원의 66% 늘어난 것으로 추산된다
 배민을 운영하는 우아한형제들 역시 지난해 매출이 1조995억원으로 전년 대비 94
4% 늘었다
쿠팡은 특히 식료품 판매가 늘었다
 신선식품 등 먹거리를 오프라인 대신 온라인으로 구매하는 수요가 급격하게 증가하면서 쿠팡이 수혜를 누린 것으로 분석된다
 실제로 지난해 온라인 식품시장 거래액은 43조4000억원으로 전년보다 62
4% 성장했다
배민도 음식 배달시장 성장과 맞물렸다
 지난해 음식서비스의 온라인 거래액은 17조4000억원으로 전년 대비 78
6% 늘었다
 올 1분기에도 온라인 음식서비스 거래액은 지난해 같은 기간 대비 71
9% 늘며 1분기 전체 온라인쇼핑 거래액 신장률(21
3%)을 웃돌았다
 업계 관계자는 "코로나19 유행 기간에 높은 성장세를 보인 서비스 대부분이 한 번에 지출하는 비용보다 결제건수가 꾸준히 증가했다"면서 "1인당 결제 빈도가 늘었다는 것은 더 장기적인 성과를 기대할 수 있게 한다"고 말했다
박준호기자 junho@etnews.com
35
----------print Analysis Content----------
Provider :  전자신문
Category :  연예
ContentLen :  1436
ContentNumericalCitationNum :  57
AverageSentenceLen :  40.857142857142854
AverageAdverbSentenceNum :  0.34285714285714286
AverageQuotesNum :  2
AverageQuotesLen :  0.07172701949860724
QuotesBuffer :  "코로나19 유행 기간에 높은 성장세를 보인 서비스 대부분이 한 번에 지출하는 비용보다 결제건수가 꾸준히 증가했다"
QuotesBuffer :  "1인당 결제 빈도가 늘었다는 것은 더 장기적인 성과를 기대할 수 있게 한다"
----------End Analysis Content----------
----------print Value----------
BylineScore :  1
ContentLenScore :  0.835
QuotesNumScore :  0.13333333333333333
TitleLenScore :  0
TitleQuestionMarkExclamationMarkScore :  0
ContentNumericalCitationNumScore :  1
AverageSentenceLenScore :  0
TitleAdverbsCountScore :  0
AverageAdverbSentenceNumScore :  0
AverageQuotesLenScore :  0
Journal :  {'scoreTotal': 73.18262166666666, 'readability': 1.3576383333333335, 'transparency': 9.059505, 'factuality': 8.386805, 'utility': 8.63763, 'fairness': 5.9026700000000005, 'diversity': 7.454456666666666, 'originality': 10.53462, 'importance': 6.888671666666666, 'depth': 10.469624999999999, 'sensationalism': 4.491}
Vanilla :  {'scoreTotal': 26.715, 'readability': 2.9683333333333333, 'transparency': 2.9683333333333333, 'factuality': 2.9683333333333333, 'utility': 2.9683333333333333, 'fairness': 1.9683333333333333, 'diversity': 2.9683333333333333, 'originality': 2.9683333333333333, 'importance': 2.9683333333333333, 'depth': 2.9683333333333333, 'sensationalism': 1.0}
----------End Value----------
----------print Analysis Title----------
TitleText :  다온플래닛, 인프라 모니터링 서비스 지원
TitleMecabTag :  [('다', 'MAG'), ('온', 'VV+ETM'), ('플래닛', 'NNP'), (',', 'SC'), ('인프라', 'NNG'), ('모니터링', 'NNG'), ('서비스', 'NNG'), ('지원', 'NNG')]
TitleLen :  22
QuestionMarkCount :  0
ExclamationMarkCount :  0
TitleAdverbsCount :  1
TitleDoubleQuotationsMarksNum :  0
----------End Analysis Title----------
13
다온플래닛(대표 임종화)이 인프라 모니터링 서비스 'DIMS(Daonplanet Infra Monitoring Service)'를 출시하고 기업 인프라 시장 공략에 나섰다
 회사는 자체 개발한 모니터링 프로그램과 협력사 프로그램을 결합해 중소기업을 대상으로 솔루션 공급에 나선다
DIMS는 간이 네트워크 관리 프로토콜(SNMP)과 VPN 터널링을 기반으로 하는 인프라 모니터링 서비스다
 VPN 터널링은 VPN으로 연결한 두 지점 간에 가상 통로를 생성하는 것이다
 이 과정에서 사설망과 같은 보안 기능을 제공한다
서비스는 기업 내부에서 쓰이는 △방화벽 △스위치 △서버 △PC △무선컨트롤러 △모바일 AP를 대상으로 로그 데이터와 여러 정보를 취합한 네트워크 보안 리포트를 고객사에 제공한다
 인프라 장애가 발생할 경우 자동으로 고객사 메일과 소셜미디어를 통해 알림 정보를 제공한다
 고객사에 관련 유지 보수 서비스도 제공한다
 솔루션 적용에 별도 통합보안관리(ESM)와 네트워크 관리 시스템(NMS) 구축이 필요없다
임종화 대표는 "기업의 올바른 인프라 네트워크 구성을 지원할 것"이라며 "중소기업 기업 환경에 최적화한 네트워크 보안 환경을 제공할 계획"이라고 말했다
 이어 "가용성과 안정성, 효율성을 잡을 수 있는 최적 솔루션이라고 자부한다"고 덧붙였다
임중권기자 lim9181@etnews.com
12
----------print Analysis Content----------
Provider :  전자신문
Category :  연예
ContentLen :  677
ContentNumericalCitationNum :  1
AverageSentenceLen :  56.166666666666664
AverageAdverbSentenceNum :  0.16666666666666666
AverageQuotesNum :  4
AverageQuotesLen :  0.20531757754800592
QuotesBuffer :  'DIMS(Daonplanet Infra Monitoring Service)'
QuotesBuffer :  "기업의 올바른 인프라 네트워크 구성을 지원할 것"
QuotesBuffer :  "중소기업 기업 환경에 최적화한 네트워크 보안 환경을 제공할 계획"
QuotesBuffer :  "가용성과 안정성, 효율성을 잡을 수 있는 최적 솔루션이라고 자부한다"
----------End Analysis Content----------
----------print Value----------
BylineScore :  1
ContentLenScore :  0.165
QuotesNumScore :  0.26666666666666666
TitleLenScore :  0
TitleQuestionMarkExclamationMarkScore :  0
ContentNumericalCitationNumScore :  0
AverageSentenceLenScore :  0
TitleAdverbsCountScore :  -0.5
AverageAdverbSentenceNumScore :  0
AverageQuotesLenScore :  0
Journal :  {'scoreTotal': 41.02127833333334, 'readability': -1.2312383333333334, 'transparency': 6.193495, 'factuality': 5.754595, 'utility': 4.60477, 'fairness': 4.040830000000001, 'diversity': 2.4889433333333333, 'originality': 6.16878, 'importance': 4.006328333333333, 'depth': 6.253775000000001, 'sensationalism': 2.7409999999999997}
Vanilla :  {'scoreTotal': 11.885, 'readability': 0.9316666666666666, 'transparency': 1.4316666666666666, 'factuality': 0.9316666666666666, 'utility': 1.4316666666666666, 'fairness': 0.9316666666666666, 'diversity': 1.4316666666666666, 'originality': 1.4316666666666666, 'importance': 1.4316666666666666, 'depth': 1.4316666666666666, 'sensationalism': 0.5}
----------End Value----------
----------print Analysis Title----------
TitleText :  '메탄올 뿌려서 소독?'…또 다른 재난 '인포데믹' 피하려면?
TitleMecabTag :  [("'", 'SY'), ('메탄올', 'NNG'), ('뿌려서', 'VV+EC'), ('소독', 'NNG'), ('?', 'SF'), ("'…", 'SY'), ('또', 'MAG'), ('다른', 'MM'), ('재난', 'NNG'), ("'", 'SY'), ('인포', 'NNP'), ('데', 'NNB'), ('믹', 'NNG'), ("'", 'SY'), ('피하', 'VV'), ('려면', 'EF'), ('?', 'SF')]
TitleLen :  34
QuestionMarkCount :  2
ExclamationMarkCount :  0
TitleAdverbsCount :  1
TitleDoubleQuotationsMarksNum :  0
----------End Analysis Title----------
63
63
----------print Analysis Content----------
Provider :  전자신문
Category :  연예
ContentLen :  2515
ContentNumericalCitationNum :  23
AverageSentenceLen :  38.98412698412698
AverageAdverbSentenceNum :  0.36507936507936506
AverageQuotesNum :  5
AverageQuotesLen :  0.11928429423459244
QuotesBuffer :  "메탄올 소독에 사용하면 위험"
QuotesBuffer :  "코로나19 방역을 위해 메탄올을 사용해서는 안된다"
QuotesBuffer :  "메탄올은 인화성이 강한 무색 액체로 반복 노출되면 중추신경계와 시신경에 손상을 유발하는 독성 물질"
QuotesBuffer :  '은혜의 강 교회'
QuotesBuffer :  "사실상 직접 접촉과 다름없어"
QuotesBuffer :  "예배 참석자 중에 확진자가 있었고 분무기를 소독하지 않고 계속 뿌렸기 때문에 사실상의 직접접촉이나 다름이 없다고 추정한다"
QuotesBuffer :  "잘못된 정보로 인한 인포데믹(Infomemic, 정보감염증)"
QuotesBuffer :  "코로나19에 대한 정보는 신뢰할 수 있는 기관에서 얻어야 한다.⏎
친구의, 친구의, 친구의, 친구의 이웃(their "
QuotesBuffer :  's friend'
QuotesBuffer :  's friend'
----------End Analysis Content----------
----------print Value----------
BylineScore :  1
ContentLenScore :  1
QuotesNumScore :  0.3333333333333333
TitleLenScore :  0
TitleQuestionMarkExclamationMarkScore :  -1
ContentNumericalCitationNumScore :  1
AverageSentenceLenScore :  0
TitleAdverbsCountScore :  -0.5
AverageAdverbSentenceNumScore :  0
AverageQuotesLenScore :  0
Journal :  {'scoreTotal': 67.30016666666667, 'readability': -0.8766666666666667, 'transparency': 6.836000000000001, 'factuality': 9.414000000000001, 'utility': 9.615, 'fairness': 5.2465, 'diversity': 8.778666666666666, 'originality': 8.886000000000001, 'importance': 8.166666666666668, 'depth': 11.994, 'sensationalism': -0.7600000000000002}
Vanilla :  {'scoreTotal': 22.0, 'readability': 1.8333333333333335, 'transparency': 2.3333333333333335, 'factuality': 1.8333333333333335, 'utility': 3.3333333333333335, 'fairness': 0.8333333333333335, 'diversity': 3.3333333333333335, 'originality': 2.3333333333333335, 'importance': 3.3333333333333335, 'depth': 3.3333333333333335, 'sensationalism': -0.5}
----------End Value----------

```