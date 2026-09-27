<script setup lang="ts">
// 四段式应用外壳:导航轨 + 上下文栏 + 弹性主区(第四段右侧抽屉由对话模块接入)。
// 响应式:宽屏(>900px)上下文栏为常驻列;窄屏收起为抽屉,由左上角「»」唤出、遮罩点击收回。
import { onBeforeUnmount, onMounted, ref } from 'vue'

import ContextPanel from './ContextPanel.vue'
import NavRail from './NavRail.vue'

const NARROW_QUERY = '(max-width: 900px)'

const panelCollapsed = ref(false)
/** 窄屏标记:决定上下文栏按抽屉渲染 */
const narrow = ref(false)

let mq: MediaQueryList | null = null

function onMediaChange(e: MediaQueryListEvent): void {
  narrow.value = e.matches
  // 断点切换时折叠态跟随形态:窄屏默认收起,宽屏默认展开
  panelCollapsed.value = e.matches
}

onMounted(() => {
  mq = window.matchMedia(NARROW_QUERY)
  narrow.value = mq.matches
  panelCollapsed.value = mq.matches
  mq.addEventListener('change', onMediaChange)
})

onBeforeUnmount(() => {
  mq?.removeEventListener('change', onMediaChange)
})
</script>

<template>
  <div class="shell" :class="{ 'shell--collapsed': panelCollapsed }">
    <NavRail class="shell__rail" />
    <ContextPanel v-show="!panelCollapsed" class="shell__panel" @collapse="panelCollapsed = true" />

    <!-- 窄屏抽屉的遮罩:点击收回上下文栏 -->
    <div
      v-if="narrow && !panelCollapsed"
      class="shell__scrim"
      aria-hidden="true"
      @click="panelCollapsed = true"
    />

    <main class="shell__main">
      <!-- 主区:四视图懒加载 + KeepAlive(切视图不卸载,保住对话状态,UI-DESIGN §1.2) -->
      <router-view v-slot="{ Component }">
        <KeepAlive>
          <component :is="Component" />
        </KeepAlive>
      </router-view>

      <!-- 折叠后用于恢复上下文栏 -->
      <button
        v-if="panelCollapsed"
        type="button"
        class="shell__restore"
        title="展开上下文栏"
        aria-label="展开上下文栏"
        @click="panelCollapsed = false"
      >
        »
      </button>
    </main>
  </div>
</template>

<style scoped>
.shell {
  display: grid;
  grid-template-columns: var(--layout-rail) var(--layout-panel) minmax(0, 1fr);
  height: 100%;
}

.shell--collapsed {
  grid-template-columns: var(--layout-rail) 0 minmax(0, 1fr);
}

/* 显式列定位:防止 ContextPanel 被 v-show 移除后,主区被自动排进 0 宽的折叠列 */
.shell__rail {
  grid-column: 1;
}

.shell__panel {
  grid-column: 2;
}

.shell__main {
  grid-column: 3;
  position: relative;
  min-width: 0;
  overflow-y: auto;
  background: var(--bg-base);
}

.shell__restore {
  position: absolute;
  top: var(--sp-3);
  left: var(--sp-3);
  z-index: 20;
  width: 28px;
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-raised);
  border: 1px solid var(--border-subtle);
  border-radius: var(--r-md);
  box-shadow: var(--shadow-card);
  color: var(--text-secondary);
  font-size: var(--fs-sm);
  cursor: pointer;
  transition:
    background var(--t-fast),
    color var(--t-fast);
}

.shell__restore:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

.shell__scrim {
  display: none;
}

/* 窄屏:上下文栏改为左侧抽屉,主区占满 */
@media (max-width: 900px) {
  .shell,
  .shell--collapsed {
    grid-template-columns: var(--layout-rail) minmax(0, 1fr);
  }

  .shell__main {
    grid-column: 2;
  }

  .shell__panel {
    position: fixed;
    top: 0;
    bottom: 0;
    left: var(--layout-rail);
    z-index: 40;
    width: min(var(--layout-panel), calc(100vw - var(--layout-rail) - 48px));
    box-shadow: var(--shadow-pop);
  }

  .shell__scrim {
    position: fixed;
    inset: 0;
    left: var(--layout-rail);
    z-index: 39;
    display: block;
    background: rgba(15, 17, 21, 0.4);
  }
}
</style>
