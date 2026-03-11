agas run-episode \
  --processed-root processed \
  --dataset ml-latest-small \
  --target-item-id 101 \
  --target-keyword horror \
  --num-agents 4 \
  --num-steps 4 \
  --coordinator-policy openai \
  --llm-model gpt-5-mini \
  --output outputs/episode_result.json