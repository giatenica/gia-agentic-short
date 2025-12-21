

## Architecture

- **Claude 4.5 Family** via Anthropic API with task-based model selection
- **Cloudflare** Workers, KV, R2, D1 available for edge compute and storage
- **Microsoft Azure** available for additional cloud services

## Claude API Best Practices
1. **No hard token limits**: Let Claude use full context for best results
2. **Prompt caching**: Always cache system prompts and repeated context (90% cost savings on cache hits)
3. **Efficient prompts**: Be specific, provide examples, use structured output formats
4. **Batch processing**: Use for non-urgent bulk tasks (50% cost savings)
5. **Extended thinking**: Enable for complex reasoning tasks (budget 16k+ tokens)

## Agent Model Configuration
| Task Type | Model | Use Case |
|-----------|-------|----------|
| Complex Reasoning | `claude-opus-4-5-20251101` | Research, scientific analysis, academic writing |
| Coding/Agents | `claude-sonnet-4-5-20250929` | Default for most tasks, agents, data analysis |
| High-Volume | `claude-haiku-4-5-20251001` | Classification, summarization, extraction |
| Fallback | `openai/gpt-4.1-mini` | GitHub Models backup |

## Cloud Infrastructure Available
| Service | Provider | Use Case |
|---------|----------|----------|
| Workers | Cloudflare | Edge compute, API endpoints, scheduled tasks |
| KV | Cloudflare | Key-value storage, caching, session data |
| R2 | Cloudflare | Object storage (S3-compatible), large files |
| D1 | Cloudflare | SQLite database at edge |
| Azure | Microsoft | Additional compute, storage, AI services |

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
- Always gitignore docs folder and .github 
- Always maintain a clear and clean file structure, remove redundant and temporary files and code if not needed anymore
- The Author of all academic Writing is always Gia Tenica (me@giatenica.com)
- Use an Asterisks (*) after her name with the info in the footnote Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher, for more information see: https://giatenica.com

