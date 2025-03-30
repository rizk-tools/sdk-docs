import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const SOURCE_DIR = path.join(__dirname, 'markdown-source'); // Your existing markdown files
const TARGET_DIR = path.join(__dirname, 'src/content/docs');

// Mapping for sections
const SECTION_MAPPING = {
  'README.md': { targetPath: 'index.md' },
  'docs/api': 'api',
  'docs/core-concepts': 'core-concepts',
  'docs/getting-started': 'getting-started'
};

// Function to make directory recursively
async function ensureDir(dir) {
  try {
    await fs.mkdir(dir, { recursive: true });
  } catch (error) {
    if (error.code !== 'EEXIST') {
      throw error;
    }
  }
}

// Extract title from markdown content
function extractTitle(content) {
  const titleMatch = content.match(/^#\s+(.+)$/m);
  return titleMatch ? titleMatch[1].trim() : '';
}

// Extract description from markdown content
function extractDescription(content) {
  const paragraphMatch = content.match(/^#\s+.+\n\n(.+?)(?:\n\n|$)/s);
  return paragraphMatch ? paragraphMatch[1].trim().substring(0, 160) : '';
}

// Process a single markdown file
async function processMarkdownFile(sourcePath, targetPath) {
  try {
    // Read the source file
    const content = await fs.readFile(sourcePath, 'utf-8');
    
    // Check if it already has frontmatter
    const hasFrontmatter = content.startsWith('---');
    
    let processedContent;
    
    if (hasFrontmatter) {
      // File already has frontmatter, we'll use it as is
      processedContent = content;
    } else {
      // Create frontmatter
      const title = extractTitle(content) || path.basename(sourcePath, '.md');
      const description = extractDescription(content) || `Documentation for ${title}`;
      
      // Create the frontmatter
      const frontmatter = `---
title: ${JSON.stringify(title)}
description: ${JSON.stringify(description)}
---

${content}`;
      
      processedContent = frontmatter;
    }
    
    // Ensure the target directory exists
    await ensureDir(path.dirname(targetPath));
    
    // Write the file
    await fs.writeFile(targetPath, processedContent);
    console.log(`Processed: ${sourcePath} -> ${targetPath}`);
  } catch (error) {
    console.error(`Error processing file ${sourcePath}:`, error);
  }
}

// Process directory and its subdirectories
async function processDirectory(baseSourceDir, currentSourceDir, currentTargetDir) {
  try {
    // Get all files and directories in the current source directory
    const entries = await fs.readdir(currentSourceDir, { withFileTypes: true });
    
    for (const entry of entries) {
      const sourcePath = path.join(currentSourceDir, entry.name);
      const relativePath = path.relative(baseSourceDir, sourcePath);
      
      // Check for special mappings
      let targetDir = currentTargetDir;
      let targetFileName = entry.name;
      
      // Handle special cases
      for (const [key, value] of Object.entries(SECTION_MAPPING)) {
        if (relativePath === key) {
          if (typeof value === 'object' && value.targetPath) {
            targetFileName = value.targetPath;
            break;
          }
        } else if (relativePath.startsWith(`${key}/`)) {
          if (typeof value === 'string') {
            targetDir = path.join(TARGET_DIR, value);
            targetFileName = relativePath.substring(key.length + 1);
            break;
          }
        }
      }
      
      if (entry.isDirectory()) {
        // Process the subdirectory
        await processDirectory(
          baseSourceDir,
          sourcePath,
          path.join(targetDir, entry.name)
        );
      } else if (entry.name.endsWith('.md')) {
        // Handle README.md specially for directories
        if (entry.name === 'README.md') {
          // If we're in a non-root directory, make it the index
          if (currentSourceDir !== baseSourceDir) {
            const parentDir = path.basename(currentSourceDir);
            await processMarkdownFile(
              sourcePath,
              path.join(path.dirname(targetDir), `${parentDir}/index.md`)
            );
            continue;
          }
        }
        
        // Remove .md extension and replace with Astro's directory-based routing
        const baseName = entry.name === 'README.md' ? 'index' : path.basename(entry.name, '.md');
        
        // Create target path
        let targetPath;
        if (baseName === 'index') {
          targetPath = path.join(targetDir, 'index.md');
        } else {
          targetPath = path.join(targetDir, baseName, 'index.md');
        }
        
        await processMarkdownFile(sourcePath, targetPath);
      }
    }
  } catch (error) {
    console.error(`Error processing directory ${currentSourceDir}:`, error);
  }
}

// Main function
async function main() {
  try {
    console.log(`Converting markdown files from ${SOURCE_DIR} to ${TARGET_DIR}...`);
    
    // Ensure the target directory exists
    await ensureDir(TARGET_DIR);
    
    // Process the root directory
    await processDirectory(SOURCE_DIR, SOURCE_DIR, TARGET_DIR);
    
    console.log('Conversion complete!');
  } catch (error) {
    console.error('Error converting markdown files:', error);
    process.exit(1);
  }
}

main();