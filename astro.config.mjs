import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://docs.rizk.tools',
  integrations: [
    starlight({
      title: 'Documentation',
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
            { label: 'Installation', link: '/getting-started/installation/' },
            { label: 'Quickstart', link: '/getting-started/quickstart/' },
            { label: 'Configuration', link: '/getting-started/configuration/' },
          ],
        },
        {
          label: 'Core Concepts',
          items: [
            { label: 'Guardrails', link: '/core-concepts/guardrails/' },
            { label: 'Policy Enforcement', link: '/core-concepts/policy-enforcement/' },
            { label: 'Telemetry', link: '/core-concepts/telemetry/' },
            { label: 'Tracing', link: '/core-concepts/tracing/' },
          ],
        },
        {
          label: 'API Reference',
          items: [
            { label: 'Rizk Class', link: '/api/rizk/' },
            { label: 'Client', link: '/api/client/' },
            { label: 'Fast Rules Engine', link: '/api/fast-rules-engine/' },
            { label: 'Guardrails Engine', link: '/api/guardrails-engine/' },
            { label: 'LLM Fallback', link: '/api/llm-fallback/' },
            { label: 'Policy Augmentation', link: '/api/policy-augmentation/' },
            { label: 'State Manager', link: '/api/state-manager/' },
            { label: 'Telemetry', link: '/api/telemetry/' },
          ],
        },
      ],
      customCss: [
        './src/styles/custom.css',
      ],
    }),
  ],
});