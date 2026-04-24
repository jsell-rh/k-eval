# Changelog

## [1.2.1](https://github.com/jsell-rh/k-eval/compare/v1.2.0...v1.2.1) (2026-04-24)


### Bug Fixes

* add checkout step before uv.lock sync in release workflow ([#38](https://github.com/jsell-rh/k-eval/issues/38)) ([dff1435](https://github.com/jsell-rh/k-eval/commit/dff1435a5a8cdd119b48e3549364255c4b65b9a2))
* cleanup diagnostic infrastructure and sync uv.lock on release ([#36](https://github.com/jsell-rh/k-eval/issues/36)) ([2e5f351](https://github.com/jsell-rh/k-eval/commit/2e5f351940bf6a1b79ada16e4b8698f97e70db07))

## [1.2.0](https://github.com/jsell-rh/k-eval/compare/v1.1.3...v1.2.0) (2026-04-24)


### Features

* add allowed_tools condition config for baseline evaluations ([#33](https://github.com/jsell-rh/k-eval/issues/33)) ([aa53c66](https://github.com/jsell-rh/k-eval/commit/aa53c663e9f5ba19d6a6e9849394460d2c2b036b))

## [1.1.3](https://github.com/jsell-rh/k-eval/compare/v1.1.2...v1.1.3) (2026-03-24)


### Bug Fixes

* **sec:** pin litellm to prevent exploit ([#31](https://github.com/jsell-rh/k-eval/issues/31)) ([af7582f](https://github.com/jsell-rh/k-eval/commit/af7582f89bbd076dc2489d676d7d986105761a51))

## [1.1.2](https://github.com/jsell-rh/k-eval/compare/v1.1.1...v1.1.2) (2026-02-27)


### Bug Fixes

* treat InternalServerError as retriable in judge ([#29](https://github.com/jsell-rh/k-eval/issues/29)) ([4ad294f](https://github.com/jsell-rh/k-eval/commit/4ad294f2915be6fa755d6fa7b0de4d9da6ef7eef))

## [1.1.1](https://github.com/jsell-rh/k-eval/compare/v1.1.0...v1.1.1) (2026-02-27)


### Bug Fixes

* release viewer copy button polish ([#27](https://github.com/jsell-rh/k-eval/issues/27)) ([1e6c4ac](https://github.com/jsell-rh/k-eval/commit/1e6c4ac9a0f4e2e774f344a5d71bde81b150d81d))

## [1.1.0](https://github.com/jsell-rh/k-eval/compare/v1.0.0...v1.1.0) (2026-02-27)


### Features

* mcp reliability and agent traceability ([#24](https://github.com/jsell-rh/k-eval/issues/24)) ([c5807fd](https://github.com/jsell-rh/k-eval/commit/c5807fd5429df04e3e84fd64427daaa1f0ebe6bd))

## [1.0.0](https://github.com/jsell-rh/k-eval/compare/v0.3.2...v1.0.0) (2026-02-26)


### ⚠ BREAKING CHANGES

* add interactive results viewer ([#20](https://github.com/jsell-rh/k-eval/issues/20))

### Features

* add interactive results viewer ([#20](https://github.com/jsell-rh/k-eval/issues/20)) ([316a431](https://github.com/jsell-rh/k-eval/commit/316a431a1e2574daf4451cca475aa7b680ece10d))

## [0.3.2](https://github.com/jsell-rh/k-eval/compare/v0.3.1...v0.3.2) (2026-02-26)


### Documentation

* modify k-eval command examples in README ([#18](https://github.com/jsell-rh/k-eval/issues/18)) ([2126036](https://github.com/jsell-rh/k-eval/commit/21260362cfad427884dda2ad8e665f4ab0a29769))

## [0.3.1](https://github.com/jsell-rh/k-eval/compare/v0.3.0...v0.3.1) (2026-02-25)


### Documentation

* add instructions for running directly from pypi ([#16](https://github.com/jsell-rh/k-eval/issues/16)) ([4c48a6f](https://github.com/jsell-rh/k-eval/commit/4c48a6fd611777c406c1ca952b0d8c0dff550cc1))

## [0.3.0](https://github.com/jsell-rh/k-eval/compare/v0.2.0...v0.3.0) (2026-02-25)


### Features

* add CLI ([#9](https://github.com/jsell-rh/k-eval/issues/9)) ([cb1a723](https://github.com/jsell-rh/k-eval/commit/cb1a7236bd767e47f9adbcbe98c09e46aeb31c2f))
* **ci:** ci checks & pypi publishing ([#12](https://github.com/jsell-rh/k-eval/issues/12)) ([d775b0f](https://github.com/jsell-rh/k-eval/commit/d775b0faa2f7d15c9f4797e27dd40636a387161f))
* cli ([#8](https://github.com/jsell-rh/k-eval/issues/8)) ([8af758b](https://github.com/jsell-rh/k-eval/commit/8af758b7e42b484c8b8f5481b3a32ef479e1b728))
* config loader ([#3](https://github.com/jsell-rh/k-eval/issues/3)) ([468fb50](https://github.com/jsell-rh/k-eval/commit/468fb5086d7122a39f303319cd4c72a92edbf335))
* dataset loading ([#4](https://github.com/jsell-rh/k-eval/issues/4)) ([aaab556](https://github.com/jsell-rh/k-eval/commit/aaab5564219d399c287a6e19c42b96e5cbda577f))
* implement agent domain, AgentResult, and ClaudeAgentSDKAgent with tool whitelisting ([#5](https://github.com/jsell-rh/k-eval/issues/5)) ([cd08249](https://github.com/jsell-rh/k-eval/commit/cd0824995bc799b6d834578480291b8c5473f266))
* implement evaluation concurrency ([#10](https://github.com/jsell-rh/k-eval/issues/10)) ([e2c0ce1](https://github.com/jsell-rh/k-eval/commit/e2c0ce1222aeeb879a1d5cbc3a1ebf9e1a926b74))
* implement evaluation runner ([#7](https://github.com/jsell-rh/k-eval/issues/7)) ([81deeb8](https://github.com/jsell-rh/k-eval/commit/81deeb8324ced75393be91c369459dcdc3fac921))
* implement lllm judge ([#6](https://github.com/jsell-rh/k-eval/issues/6)) ([f29472a](https://github.com/jsell-rh/k-eval/commit/f29472ad83e06bbf9026c717e3b3309dc51f269d))


### Bug Fixes

* **cd:** fix pypi release ([#14](https://github.com/jsell-rh/k-eval/issues/14)) ([88dcbc4](https://github.com/jsell-rh/k-eval/commit/88dcbc4e50bef5c4bf87410d1d5c96d3bb65b795))

## [0.2.0](https://github.com/jsell-rh/k-eval/compare/k-eval-v0.1.0...k-eval-v0.2.0) (2026-02-25)


### Features

* add CLI ([#9](https://github.com/jsell-rh/k-eval/issues/9)) ([cb1a723](https://github.com/jsell-rh/k-eval/commit/cb1a7236bd767e47f9adbcbe98c09e46aeb31c2f))
* **ci:** ci checks & pypi publishing ([#12](https://github.com/jsell-rh/k-eval/issues/12)) ([d775b0f](https://github.com/jsell-rh/k-eval/commit/d775b0faa2f7d15c9f4797e27dd40636a387161f))
* cli ([#8](https://github.com/jsell-rh/k-eval/issues/8)) ([8af758b](https://github.com/jsell-rh/k-eval/commit/8af758b7e42b484c8b8f5481b3a32ef479e1b728))
* config loader ([#3](https://github.com/jsell-rh/k-eval/issues/3)) ([468fb50](https://github.com/jsell-rh/k-eval/commit/468fb5086d7122a39f303319cd4c72a92edbf335))
* dataset loading ([#4](https://github.com/jsell-rh/k-eval/issues/4)) ([aaab556](https://github.com/jsell-rh/k-eval/commit/aaab5564219d399c287a6e19c42b96e5cbda577f))
* implement agent domain, AgentResult, and ClaudeAgentSDKAgent with tool whitelisting ([#5](https://github.com/jsell-rh/k-eval/issues/5)) ([cd08249](https://github.com/jsell-rh/k-eval/commit/cd0824995bc799b6d834578480291b8c5473f266))
* implement evaluation concurrency ([#10](https://github.com/jsell-rh/k-eval/issues/10)) ([e2c0ce1](https://github.com/jsell-rh/k-eval/commit/e2c0ce1222aeeb879a1d5cbc3a1ebf9e1a926b74))
* implement evaluation runner ([#7](https://github.com/jsell-rh/k-eval/issues/7)) ([81deeb8](https://github.com/jsell-rh/k-eval/commit/81deeb8324ced75393be91c369459dcdc3fac921))
* implement lllm judge ([#6](https://github.com/jsell-rh/k-eval/issues/6)) ([f29472a](https://github.com/jsell-rh/k-eval/commit/f29472ad83e06bbf9026c717e3b3309dc51f269d))
