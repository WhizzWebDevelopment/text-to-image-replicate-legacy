# Output Directory

Generated images are saved here.

## Running Examples

```bash
# Set up credentials
export REPLICATE_API_TOKEN="r8_..."

# Run example generations
bash _output/examples/run_replicate_examples.sh
```

## Example Outputs

The example script generates 7 images:
1. **Landscape** - Basic SDXL generation (1024×1024)
2. **Portrait** - With negative prompt
3. **Abstract** - High guidance scale
4. **Robot** - Fast SD 1.5 generation (512×512)
5. **Dragon** - 4 variations at once
6. **City** - Reproducible with seed
7. **Banner** - Wide format (1536×512)

**Cost**: ~$0.14 (or use free tier credits!)
