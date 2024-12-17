# Score Vision (SN44)

Score Vision is a decentralized computer vision framework built on Bittensor, focusing on Game State Recognition (GSR) in football matches. Our framework enables complex computer vision tasks through lightweight validation, addressing the significant untapped potential in football video analysis.

## Overview

Traditional video annotation costs range from $10-55 per minute, with complex sports scenarios requiring up to 4 hours of human labeling time per minute. A single football match (90+ minutes) requires approximately 360 hours of manual annotation work, costing $1,000-5,000 for comprehensive labeling.

Score Vision addresses these challenges through:

- Decentralized computation for video processing
- Lightweight validation using Vision Language Models (VLMs) and Human-in-the-loop
- Carefully designed incentive mechanisms
- Alignment with the SoccerNet-GSR framework

## Architecture

The system operates with three primary roles:

1. **Miners**: Process video streams using computer vision models

   - Handle object detection and tracking
   - Generate standardized outputs
   - Implement custom optimization strategies

2. **Validators**: Verify miners' outputs efficiently

   - Use selective frame analysis
   - Employ VLMs for accuracy assessmentx
   - Maintain network integrity

3. **Subnet Owners**: Manage network parameters
   - Oversee incentive mechanisms
   - Adjust system parameters
   - Ensure network adaptability

## Setup Instructions

- [Miner Setup Guide](miner/README.md)
- [Validator Setup Guide](validator/README.md)

## Technical Implementation

### Validation Mechanism

> **Note:** Our validation mechanism is being constantly improved and tweaked during testnet phase especially.

Our framework uses a novel two-phase validation approach:

1. **Object Count Verification**

   - Numerical analysis of scene composition
   - Structured JSON output with precise counts
   - Baseline establishment for quality evaluation

2. **Bounding Box Quality Assessment**
   - Completeness verification
   - Spatial accuracy examination
   - Normalized quality scoring (0-1)

### Frame Sampling

- Random frame selection
- Unpredictable validation patterns
- Efficient computational resource usage

### Performance Metrics

(Coming soon)

The Game State Higher Order Tracking Accuracy (GS-HOTA) metric:

```
GS-HOTA = √(Detection × Association)
```

Where:

- Detection: Measures object detection accuracy
- Association: Assesses tracking consistency

## Roadmap

### Phase 1 (Current)

- [x] Game State Recognition challange implementation
- [x] VLM-based validation
- [x] Incentive mechanism
- [x] Testnet deploy on netuid 261
- [ ] Community testing
- [ ] Comprehensive benchmarking

### Phase 2 (Q1 2025)

- [ ] Human-in-the-loop validation
- [ ] Additional footage type (grassroots)
- [ ] Dashboard and Leaderboard

### Phase 3 (Q2-Q3 2025)

- [ ] Action spotting integration
- [ ] Match event captioning
- [ ] Advanced player tracking

### Phase 4 (Q4 2025)

- [ ] Integration APIs
- [ ] Additional sports adaptation
- [ ] Developer tools and SDKs
- [ ] Community contribution framework

## Future Developments

### Action Spotting and Captioning

- Event detection (goals, fouls, etc.)
- Automated highlight generation
- Natural language descriptions

### Cross-Domain Applications

- Basketball and tennis analysis
- Security surveillance
- Retail analytics

### Technical Enhancements

- Advanced VLM capabilities
- Improved attribute assessment
- Adaptive learning mechanisms
- Open-source VLM development

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Pull request process
- Development workflow
- Testing requirements

## Research

This implementation is based on our research paper:
"Score Vision: Enabling Complex Computer Vision Through Lightweight Validation - A Game State Recognition Framework for Live Football"

For technical details and methodology, please refer to the [whitepaper](https://drive.google.com/file/d/1oADURxxIZK0mTEqJPDudgXypohtFNkON/view).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
