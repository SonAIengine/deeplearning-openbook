#!/usr/bin/env bash
# PostToolUse hook: .md 파일 수정 후 GitHub KaTeX에서 깨질 가능성이 높은 패턴 검사.
# 깨짐 원인: GitHub 마크다운이 인라인 수식 $...$ 안의 _를 italic 매칭에 사용하므로,
#  - 같은 단락에 underscore 든 인라인 수식이 여러 개면 두 _ 사이가 italic으로 변환되어 KaTeX가 받기 전에 텍스트가 깨진다.
#  - 인라인의 \{...\}_{...} 패턴은 escape backslash가 마크다운에서 소실되어 깨진다.
# stderr + exit 2 로 Claude에게 피드백 (PostToolUse: 작업은 이미 끝났지만 후속 수정 유도).

set -euo pipefail

input=$(cat)

# tool_input.file_path 추출 (jq 없으면 sed fallback)
if command -v jq >/dev/null 2>&1; then
  file_path=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
else
  file_path=$(printf '%s' "$input" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
fi

[[ -n "${file_path:-}" ]] || exit 0
[[ "$file_path" =~ \.md$ ]] || exit 0
[[ -f "$file_path" ]] || exit 0

warnings=""

# 패턴 1: 한 줄에 underscore가 든 인라인 수식이 2개 이상 (italic 매칭 위험)
# 예) "$y_c$ ... $\hat{y}_c$" 한 줄
risky_multi=$(grep -nE '\$[^$]*_[^$]*\$[^$]*\$[^$]*_[^$]*\$' "$file_path" || true)
if [[ -n "$risky_multi" ]]; then
  warnings+="[KaTeX 위험] 한 줄에 underscore 든 인라인 수식이 2개 이상 — italic 매칭으로 깨짐:
$risky_multi

"
fi

# 패턴 2: 인라인 수식의 \{...\}_{...} (escape brace + subscript)
risky_brace=$(grep -nE '\$[^$]*\\\{[^$]*\\\}_' "$file_path" || true)
if [[ -n "$risky_brace" ]]; then
  warnings+="[KaTeX 위험] 인라인 수식 내 \\{...\\}_{...} — display math (\$\$...\$\$)로 분리 필요:
$risky_brace

"
fi

# 패턴 3: 숫자~숫자 등 단일 ~ 범위 표기 (GFM strikethrough로 해석되어 두 ~ 사이가 취소선)
# 코드블록 내부는 제외 (백틱 라인은 검사 생략)
risky_tilde=$(grep -nE '[A-Za-z0-9.)]~[A-Za-z0-9(]' "$file_path" | grep -v '^\s*```' || true)
if [[ -n "$risky_tilde" ]]; then
  warnings+="[GFM 위험] 단일 ~ 범위 표기 — strikethrough로 해석됨. en dash(–) 또는 hyphen(-) 사용:
$risky_tilde

"
fi

# 패턴 4: 따옴표로 감싼 인라인 수식 ("$...$") — GitHub KaTeX의 $ 경계 인식 실패
risky_quoted=$(grep -nE '"\$[^$]+\$"' "$file_path" || true)
if [[ -n "$risky_quoted" ]]; then
  warnings+="[KaTeX 위험] 따옴표로 감싼 인라인 수식 \"\$...\$\" — GitHub의 \$ 경계 인식 실패. 따옴표 제거하거나 평문 표기:
$risky_quoted

"
fi

# 패턴 5: 조건부 확률 P(...|...) 패턴 (절댓값과 구분 — 함수 인자 안의 |)
# 절댓값 |x|는 정상이지만 P(Y|X), P(A|B,C)는 \mid 권장 (의미 명확화 + 일부 환경 깨짐 회피)
risky_cond=$(grep -nE '\$[^$]*[A-Za-z]\([A-Za-z][^$|]*\|[^$|]*\)' "$file_path" || true)
if [[ -n "$risky_cond" ]]; then
  warnings+="[KaTeX 권장] 조건부 확률 표기 P(...|...) — \\mid 사용 권장 (예: P(Y \\mid X)):
$risky_cond

"
fi

if [[ -n "$warnings" ]]; then
  {
    echo "===== KaTeX 위험 패턴 감지: $file_path ====="
    printf '%s' "$warnings"
    echo "→ 수정 가이드: CLAUDE.md §2 (위험 인라인은 display math로 분리)."
  } >&2
  exit 2
fi

exit 0
