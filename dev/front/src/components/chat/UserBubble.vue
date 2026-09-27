<script setup lang="ts">
// 用户消息:右对齐气泡,限宽 76%,保留换行(UI-DESIGN §3.2)。
// 气泡给描边 + 略亮的底色,和 AI 正文拉开"谁在说话"的对比;右下角收一个小圆角作指向。
import type { UserItem } from '@/types/chat'

defineProps<{ item: UserItem }>()
</script>

<template>
  <div class="bubble" data-testid="user-bubble">{{ item.text }}</div>
</template>

<style scoped>
.bubble {
  align-self: flex-end;
  max-width: 76%;
  padding: var(--sp-3) var(--sp-4);
  /* 用户气泡带一点 accent 倾向:与 AI 中性正文一眼区分"谁在说话" */
  background: var(--accent-soft);
  border: 1px solid var(--accent-border);
  /* 右下角收口:一条隐性的"发言指向",不引入箭头装饰 */
  border-radius: var(--r-xl) var(--r-xl) var(--r-sm) var(--r-xl);
  box-shadow: var(--shadow-card);
  color: var(--text-primary);
  font-size: var(--fs-lg);
  line-height: var(--lh-base);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
</style>