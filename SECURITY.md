# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in DopamineCore, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please send an email to the project maintainers with:

1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Scope

Security concerns for this project include:

- **Reward hacking**: Agents exploiting the reward signal system to produce artificially favorable signals
- **Signal injection**: External manipulation of the reward pipeline
- **State tampering**: Unauthorized modification of persisted engine state
- **Template leakage**: Internal terminology leaking into agent-visible context

## Best Practices

When using DopamineCore in production:

- Always enable safety bounds (enabled by default)
- Monitor for unusual signal patterns
- Regularly rotate injection templates
- Validate engine state integrity after deserialization
