import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://docs.rizk.tools',
  integrations: [
    starlight({
      title: 'Rizk SDK Documentation',
      logo: {
        src: './src/assets/logo.svg',
      },
      social: {
        github: 'https://github.com/rizk-tools/rizk-sdk',
      },
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Overview', link: '/getting-started/overview/' },
            { label: 'Installation', link: '/getting-started/installation/' },
            { label: 'Quickstart', link: '/getting-started/quickstart/' },
            { label: 'First Example', link: '/getting-started/first-example/' },
            { label: 'Configuration', link: '/getting-started/configuration/' },
          ],
        },
        {
          label: 'Core Concepts',
          items: [
            { label: 'Architecture', link: '/core-concepts/architecture/' },
            { label: 'Framework Detection', link: '/core-concepts/framework-detection/' },
            { label: 'Adapters', link: '/core-concepts/adapters/' },
            { label: 'Observability', link: '/core-concepts/observability/' },
            { label: 'Guardrails Overview', link: '/core-concepts/guardrails-overview/' },
            { label: 'Decorators Overview', link: '/core-concepts/decorators-overview/' },
          ],
        },
        {
          label: 'Framework Integration',
          items: [
            { label: 'OpenAI Agents', link: '/framework-integration/openai-agents/' },
            { label: 'LangChain', link: '/framework-integration/langchain/' },
            { label: 'CrewAI', link: '/framework-integration/crewai/' },
            { label: 'LlamaIndex', link: '/framework-integration/llama-index/' },
            { label: 'LangGraph', link: '/framework-integration/langgraph/' },
            { label: 'Multi-Framework', link: '/framework-integration/multi-framework/' },
            { label: 'Custom Frameworks', link: '/framework-integration/custom-frameworks/' },
          ],
        },
        {
          label: 'Decorators',
          items: [
            { label: '@workflow', link: '/decorators/workflow/' },
            { label: '@task', link: '/decorators/task/' },
            { label: '@agent', link: '/decorators/agent/' },
            { label: '@tool', link: '/decorators/tool/' },
            { label: '@crew', link: '/decorators/crew/' },
            { label: '@guardrails', link: '/decorators/guardrails/' },
            { label: '@mcp_guardrails', link: '/decorators/mcp-guardrails/' },
            { label: 'Decorator Composition', link: '/decorators/decorator-composition/' },
            { label: 'Policies', link: '/decorators/policies/' },
          ],
        },
        {
          label: 'LLM Adapters',
          items: [
            { label: 'OpenAI', link: '/llm-adapters/openai/' },
            { label: 'OpenAI Completions', link: '/llm-adapters/openai-completions/' },
            { label: 'OpenAI Agents SDK', link: '/llm-adapters/openai-agents-sdk/' },
            { label: 'Anthropic', link: '/llm-adapters/anthropic/' },
            { label: 'Gemini', link: '/llm-adapters/gemini/' },
            { label: 'Ollama', link: '/llm-adapters/ollama/' },
            { label: 'Custom LLM', link: '/llm-adapters/custom-llm/' },
          ],
        },
        {
          label: 'Guardrails',
          items: [
            { label: 'Overview', link: '/guardrails/overview/' },
            { label: 'Using Guardrails', link: '/guardrails/using-guardrails/' },
            { label: 'Policy Enforcement', link: '/guardrails/policy-enforcement/' },
            { label: 'Configuration', link: '/guardrails/configuration/' },
            { label: 'MCP Protection', link: '/guardrails/mcp-protection/' },
            { label: 'Monitoring', link: '/guardrails/monitoring/' },
          ],
        },
        {
          label: 'Observability',
          items: [
            { label: 'Tracing', link: '/observability/tracing/' },
            { label: 'Analytics', link: '/observability/analytics/' },
            { label: 'Streaming Observability', link: '/observability/streaming-observability/' },
            { label: 'Cache Analytics', link: '/observability/cache-analytics/' },
          ],
        },
        {
          label: 'Advanced Configuration',
          items: [
            { label: 'Production Setup', link: '/advanced-config/production-setup/' },
            { label: 'Environment Variables', link: '/advanced-config/environment-variables/' },
            { label: 'Performance Tuning', link: '/advanced-config/performance-tuning/' },
            { label: 'Security', link: '/advanced-config/security/' },
            { label: 'Scaling', link: '/advanced-config/scaling/' },
          ],
        },
        {
          label: 'API Reference',
          items: [
            { label: 'API Overview', link: '/api-reference/index/' },
            { label: 'Rizk Class', link: '/api-reference/rizk-class/' },
            { label: 'Decorators API', link: '/api-reference/decorators-api/' },
            { label: 'Guardrails API', link: '/api-reference/guardrails-api/' },
            { label: 'Configuration API', link: '/api-reference/configuration-api/' },
            { label: 'Types', link: '/api-reference/types/' },
            { label: 'Utilities', link: '/api-reference/utilities/' },
          ],
        },
        {
          label: 'Troubleshooting',
          items: [
            { label: 'Common Issues', link: '/troubleshooting/common-issues/' },
            { label: 'Debugging', link: '/troubleshooting/debugging/' },
            { label: 'Performance Issues', link: '/troubleshooting/performance-issues/' },
            { label: 'Integration Issues', link: '/troubleshooting/integration-issues/' },
            { label: 'Policy Debugging', link: '/troubleshooting/policy-debugging/' },
          ],
        },
      ],
      customCss: [
        './src/styles/custom.css',
      ],
    }),
  ],
});