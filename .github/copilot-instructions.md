

## Architecture

- **Claude Sonnet 4** as primary model (claude-sonnet-4-20250514)

## Agent Model Configuration
| Agent Type | Recommended Model |
|------------|------------------|
| All Tasks | `claude-sonnet-4-20250514` (Claude Sonnet 4 via Anthropic) |
| Fallback | GitHub Models via `openai/gpt-4.1-mini` |

## Critical Rules for All Agents
1. **NEVER make up data, statistics, numbers, or facts**
2. **NEVER use emojis**
3. **NEVER use em dashes** (use semicolons, colons, or periods)
4. **ALWAYS get today's date** at the start of tasks
5. **ALWAYS use web search** for current information
6. **ALWAYS cite sources** for quantitative claims

## Banned Words (NEVER USE)
delve, realm, harness, unlock, tapestry, paradigm, cutting-edge, revolutionize,
landscape, potential, findings, intricate, showcasing, crucial, pivotal, surpass,
meticulously, vibrant, unparalleled, underscore, leverage, synergy, innovative,
game-changer, testament, commendable, meticulous, highlight, emphasize, boast,
groundbreaking, align, foster, showcase, enhance, holistic, garner, accentuate,
pioneering, trailblazing, unleash, versatile, transformative, redefine, seamless,
optimize, scalable, robust (non-statistical), breakthrough, empower, streamline,
intelligent, smart, next-gen, frictionless, elevate, adaptive, effortless,
data-driven, insightful, proactive, mission-critical, visionary, disruptive,
reimagine, agile, customizable, personalized, unprecedented, intuitive,
leading-edge, synergize, democratize, automate, accelerate, state-of-the-art,
dynamic (non-technical), reliable, efficient, cloud-native, immersive, predictive,
transparent, proprietary, integrated, plug-and-play, turnkey, future-proof,
open-ended, AI-powered, next-generation, always-on, hyper-personalized,
results-driven, machine-first, paradigm-shifting, novel, unique, utilize, impactful

## Development Guidelines
- Use async/await patterns for all agent operations
- Implement proper error handling in workflows
- Add tracing for debugging multi-agent interactions
- Test agents independently before integration
- Write chapters sequentially (phased approach) for better memory management

## Additional Info and Custom Instructions
- Always update README
- Always commit and push a version to https://github.com/giatenica/gia-agentic-short as giatenica
- Git config: user.name=giatenica, user.email=me@giatenica.com
- Always gitignore docs folder but NOT .github (workflows needed)
- Always maintain a clear and clean file structure, remove redundant and temporary files and code if not needed anymore
- The Author of all academic Writing is always Gia Tenica (me@giatenica.com)
- Use an Asterisks (*) after her name with the info in the footnote Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher, for more information see: https://giatenica.com

