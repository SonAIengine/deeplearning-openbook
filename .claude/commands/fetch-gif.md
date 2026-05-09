---
description: 학습자료(.md) 임베드용 GIF/webm/mp4를 검증된 출처에서 다운로드해 assets/gifs/에 저장하고, webm·mp4면 ffmpeg로 GIF로 변환하며, 출처·라이선스 정리 및 마크다운 임베드 스니펫까지 제안한다.
argument-hint: <URL> [filename] [--section <섹션 위치 메모>]
---

# /fetch-gif — 시각자료 가져오기

이 프로젝트(deeplearning-openbook)는 GitHub에서 직접 읽히는 학습 노트다. 본문 직관을 보강하는 짧은 GIF를 검증된 출처에서 가져와 `assets/gifs/`에 저장한다. webm/mp4를 받았다면 GIF로 변환한다.

사용자 입력: `$ARGUMENTS`

## 절차 (이 순서대로 실행)

### 1. 입력 파싱
- 첫 인자: 다운로드 URL. 없으면 사용자에게 확인 요청.
- 두 번째 인자(있으면): `assets/gifs/`에 저장될 파일명 (확장자 생략 가능). 없으면 URL 끝에서 추출하되 사람이 읽기 좋게 정리(소문자, 공백·특수문자 → `_`).

### 2. 출처 페이지 검증 (URL이 위키미디어/블로그 페이지면)
- 페이지가 직접 GIF/webm 파일이 아니라면 WebFetch로 페이지를 열어 *원본 파일 URL*과 *라이선스/작자/설명*을 추출한다.
- Wikimedia Commons는 페이지에서 author·license가 명시됨 → 반드시 캡처.
- 표준 출처 우선순위 (라이선스 안전성):
  1. **Wikimedia Commons** (CC0 / CC BY / CC BY-SA) — 가장 안전
  2. **Distill.pub** (CC BY 4.0)
  3. **저자가 명시적으로 공개한 학술/블로그 자료**
  4. **광범위 인용된 ML 표준 자료** (예: Alec Radford via Sebastian Ruder blog) — 라이선스 명시 없으면 fair use·교육 목적 + 출처 attribution 필수

### 3. 디렉토리 준비
```bash
mkdir -p assets/gifs
```

### 4. 다운로드
```bash
curl -sSL -o assets/gifs/<파일명>.<원본 확장자> "<원본 URL>"
```

### 5. 변환 (webm/mp4 → GIF만 해당)
```bash
ffmpeg -y -i assets/gifs/<input>.<webm|mp4> \
  -vf "fps=15,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
  -loop 0 assets/gifs/<파일명>.gif
```
변환 성공 시 원본 webm/mp4는 삭제(저장소 가벼움).

### 6. 결과 보고
사용자에게 다음을 보고:
- 저장된 GIF 경로와 크기 (`ls -lh assets/gifs/<파일명>.gif`)
- 출처 URL, 작자, 라이선스
- 임베드 스니펫 제안 (아래 형식)

### 7. 임베드 스니펫 (사용자에게 복붙 가능하게 출력)
```markdown
![<짧은 영문 alt text>](assets/gifs/<파일명>.gif)

> *<한국어 한두 줄 설명 — 본문 흐름과 연결>.*  
> *Source: [<페이지 제목>](<페이지 URL>), <작자>, <라이선스>.*
```

`--section` 인자가 있으면 어디에 넣으면 자연스러울지 한 줄 추천 코멘트도 덧붙인다 (실제 Edit는 사용자 확인 후 수행).

## 주의사항

- **임의로 본문 .md를 편집하지 말 것.** 사용자가 *어디에 넣을지* 명시하기 전엔 다운로드와 스니펫 제안까지만.
- 다운로드 후 파일이 사실상 손상(0 byte 또는 HTML 응답)이면 재시도 또는 사용자에게 보고. `file assets/gifs/<파일명>.gif`로 확인 가능.
- 파일이 5MB를 넘으면 `ffmpeg -vf "fps=12,scale=600:-1"` 같은 설정으로 재인코딩해 크기를 줄일지 사용자에게 묻는다.
- GIF가 이미 같은 이름으로 존재하면 덮어쓰지 말고 사용자에게 확인.
- 라이선스가 명확치 않으면 다운로드 *전에* 사용자에게 보고하고 진행 여부 확인.
