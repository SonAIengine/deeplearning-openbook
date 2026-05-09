# Deep Learning Open-Book — 작업 가이드

이 저장소는 GitHub.com에서 직접 읽히는 학습 노트(.md)다. 수식은 GitHub의 **KaTeX** 렌더러로 표시되며, GitHub 마크다운 파서가 KaTeX 도달 전에 텍스트를 변환해 버리는 함정이 있어 작성 규칙을 엄격히 지킨다.

## 1. GitHub KaTeX 깨짐 — 핵심 원인

GitHub 마크다운은 인라인 수식 `$...$` 안의 `_`도 마크다운 italic(`_text_`) 매칭에 사용한다. 그래서:

- 한 단락에 underscore가 든 인라인 수식이 **2개 이상**이면, 첫 번째 `_`와 다른 인라인 수식의 `_`가 italic으로 매칭되어 KaTeX 파싱 전에 깨짐.
- escape된 중괄호 `\{`, `\}`도 마크다운이 `\X → X`로 변환해 KaTeX가 받을 때 형태가 망가질 수 있음.

**증상 예**: `$\{(x_i, y_i)\}_{i=1}^N$` → 화면에 `${(x_i, y_i)}{i=1}^N$`로 평문 표시 (underscore와 backslash 사라짐).

이 함정을 피하기 위한 작성 규칙은 아래에 명시.

## 2. 수식 작성 규칙

### 2.1 위험 패턴 — 즉시 display math로 분리 또는 형식 변경

다음은 인라인(`$...$`)으로 쓰면 거의 확실히 깨진다.

| 패턴 | 깨짐 원인 | 처방 |
|---|---|---|
| escape brace + subscript `$\{...\}_{...}$` | 마크다운 escape 변환 + italic 매칭 | display math로 분리 |
| 한 인라인 안에 underscore 3개 이상 | italic 매칭 충돌 | display math |
| 따옴표로 감싼 인라인 `"$...$"` | GitHub `$` 경계 인식 실패 → 평문 표시 | 따옴표 제거 |
| 인라인 닫는 `$` 바로 뒤 `)` `]` `}` (예: `$O(1/T^2)$).`) | 위와 같은 경계 인식 실패 | 한국어 조사·공백·마침표를 사이에 두거나 display math |
| 조건부 표기 `P(Y\|X)` (raw `\|`) | 의미 모호 + 일부 환경 깨짐 | `P(Y \mid X)` |
| 숫자 범위 `32~512` (단일 `~`) | GFM strikethrough 매칭 → 두 `~` 사이 취소선 | `32–512` (en dash) 또는 `32-512` |

### 2.2 한 단락에 underscore 든 인라인 수식이 여러 개일 때

같은 단락에 `$..._..$` 패턴이 2개 이상 모이면 italic 매칭 위험.

- 그중 하나 이상을 display math로 빼거나
- 단독 변수(`$f_\theta$`)는 의미를 살리며 인자 추가(`$f_\theta(x)$`)해 italic 매칭 가능성 줄임
- 또는 한 군데를 평문/unicode(`f_θ`)로 대체

### 2.3 일반 표기 규칙

| 권장 | 비권장 | 이유 |
|---|---|---|
| `\to` | `\rightarrow` | 짧고 동일, 충돌 없음 |
| `x_{\text{new}}` | `x_{new}` | KaTeX가 `new`를 변수곱 italic으로 표시함 |
| `f^*: \mathcal{X} \to \mathcal{Y}` | `f^* : \mathcal{X} \rightarrow \mathcal{Y}` | 공백·연산자 단축 |
| display math 위·아래 빈 줄 1개 | (붙여 쓰기) | 블록 인식 보장 |

### 2.4 인라인 vs display 결정 흐름

```
underscore 0~1개 + 한 단락에 다른 underscore 인라인 없음 → 인라인 OK
underscore 2개 이상 OR escape brace 포함 → display math로 분리
```

### 2.5 자동 검사 hook

`.claude/hooks/check-katex.sh` (PostToolUse, matcher `Edit|Write|MultiEdit`)가 .md 수정 후 다음 패턴을 grep해 위험 발견 시 stderr + exit 2로 Claude에게 피드백한다.

| 검사 패턴 | 의미 |
|---|---|
| `\$..._...\$ ... \$..._...\$` (한 줄) | 같은 단락에 underscore 든 인라인 수식 2개 이상 |
| `\$...\\\{...\\\}_...\$` | 인라인 수식 내 escape brace + subscript |

수동 실행:
```bash
echo '{"tool_input":{"file_path":"PATH"}}' | .claude/hooks/check-katex.sh
```

### 2.6 최종 검증

hook을 통과해도 GitHub.com에서 해당 .md를 직접 열어 KaTeX 렌더링을 한 번 더 확인. local preview(VSCode 등)의 마크다운 처리는 GitHub과 다를 수 있다.

## 3. 문서 톤·구조

- 각 문서 공통: 핵심 질문 → 직관/구조 → 설계 단계 → 함정 → 확장 질문 → 한 줄 요약
- "왜·비교·설계·적용" 사고 패턴 중심. 정의 암기형 X.
- 한국어 본문 + 기술 용어 원어(commit, gradient 등) 유지.

## 4. 파일 수정 시 주의

- 학습자료를 보강할 때 기존 서사 흐름과 톤(서술적, "교수님처럼 사고하기")을 유지.
- 수식은 의미가 흐려지지 않게 — display math로 빼더라도 본문 가독성 손상 최소화.
- 표 형식(`|---|---|`)을 깰 만한 multi-line 변경은 신중히.

## 5. 시각자료 가져오기

본문 직관 보강용 시각자료는 `assets/` 아래에 저장한다.
- 동영상·동적: `assets/gifs/` (.gif)
- 정적·도식: `assets/images/` (.svg, .png)

가져올 때는 `/fetch-gif <URL> [filename]` slash command를 사용하면 다운로드·변환·출처 정리가 한 번에 된다 (webm/mp4 → GIF는 ffmpeg 자동 변환). SVG·PNG도 같은 명령으로 받지만 변환은 생략된다.

라이선스 안전 순서: Wikimedia Commons(CC0/CC BY/SA) > Distill.pub(CC BY) > 저자 명시 공개 자료 > 광범위 인용된 ML 자료(Alec Radford 등, fair use + attribution).

임베드 형식:
```markdown
![alt text](assets/gifs/file.gif)

> *한 줄 설명.*  
> *Source: [출처](URL), 작자, 라이선스.*
```

상세 절차는 `.claude/commands/fetch-gif.md` 참고.

## 5. 글로벌 규칙 import

@~/.claude/CLAUDE.md
