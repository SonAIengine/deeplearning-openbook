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

if [[ -n "$warnings" ]]; then
  {
    echo "===== KaTeX 위험 패턴 감지: $file_path ====="
    printf '%s' "$warnings"
    echo "→ 수정 가이드: CLAUDE.md §2 (위험 인라인은 display math로 분리)."
  } >&2
  exit 2
fi

exit 0
