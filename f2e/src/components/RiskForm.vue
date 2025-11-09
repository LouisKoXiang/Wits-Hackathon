<template>
  <div class="form">
    <h2>貸款風險預測</h2>
    <form @submit.prevent="submit">
      <div class="form-item">
        <label>貸款金額（美元）</label>
        <input v-model.number="form.loan_amnt" type="number" min="0" required />
      </div>

      <div class="form-item">
        <label>年收入（美元）</label>
        <input
          v-model.number="form.annual_inc"
          type="number"
          min="0"
          required
        />
      </div>

      <div class="form-item">
        <label>債務比（%）</label>
        <input
          v-model.number="form.dti"
          type="number"
          min="0"
          max="100"
          step="0.1"
          required
        />
      </div>

      <div class="form-item">
        <label>帳戶總數</label>
        <input v-model.number="form.total_acc" type="number" min="0" required />
      </div>

      <div class="form-item">
        <label>循環信貸使用率（%）</label>
        <input
          v-model.number="form.revol_util"
          type="number"
          min="0"
          max="100"
          step="0.1"
          required
        />
      </div>

      <div class="form-item">
        <label>利率（%）</label>
        <input
          v-model.number="form.int_rate"
          type="number"
          min="0"
          max="100"
          step="0.1"
          required
        />
      </div>

      <button type="submit">送出預測</button>
    </form>

    <div v-if="result" class="result">
      <p>
        「系統會根據使用者的貸款金額、年收入、債務比、利率等基本財務指標，
        透過訓練過的神經網路模型（使用 TensorFlow 建構），計算出一個
        還款成功機率。 假設這次的預測分數為 0.7，代表該客戶有約 70%
        機率能如期還款， 因此被歸類為中等風險（Medium Risk）。」
      </p>
      <p>風險等級：{{ result.risk_level }}</p>
      <p>預測分數：{{ result.risk_score.toFixed(3) }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue'
import axios from 'axios'

const form = reactive({
  loan_amnt: 25000,
  annual_inc: 100000,
  dti: 25.5,
  total_acc: 15,
  revol_util: 42.3,
  int_rate: 10.5
})

const result = ref<{ risk_level: string; risk_score: number } | null>(null)

function validateForm() {
  if (form.loan_amnt <= 0) return '貸款金額必須大於 0'
  if (form.annual_inc <= 0) return '年收入必須大於 0'
  if (form.dti < 0 || form.dti > 100) return '債務比必須介於 0-100 之間'
  if (form.total_acc < 0) return '帳戶總數不得為負數'
  if (form.revol_util < 0 || form.revol_util > 100)
    return '循環信貸使用率必須介於 0-100 之間'
  if (form.int_rate < 0 || form.int_rate > 100) return '利率必須介於 0-100 之間'
  return ''
}

async function submit() {
  const msg = validateForm()
  if (msg) return alert(msg)

  try {
    const res = await axios.post(
      import.meta.env.VITE_API_BASE_URL + '/predict',
      form
    )
    result.value = res.data
  } catch (e) {
    console.error(e)
    alert('API 呼叫失敗')
  }
}
</script>

<style scoped>
.form {
  max-width: 400px;
  margin: 40px auto;
  padding: 24px;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
.form-item {
  display: flex;
  flex-direction: column;
  margin-bottom: 12px;
}
button {
  width: 100%;
  background: #2563eb;
  color: #fff;
  padding: 10px;
  border: none;
  border-radius: 8px;
}
.result {
  margin-top: 20px;
  padding: 10px;
  background: #eef6ff;
  border-radius: 8px;
}
</style>
