COMMENT=$1
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git add --all .
git commit -m "${COMMENT}"
git push origin ${CURRENT_BRANCH}