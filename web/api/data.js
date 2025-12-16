import fs from 'fs';
import path from 'path';

export default function handler(req, res) {
  try {
    // Path to the actual data file
    const dataPath = path.join(process.cwd(), 'nvidia_score_price_dump.txt');
    
    // Check if file exists, if not, look in parent directories
    let csvData;
    
    if (fs.existsSync(dataPath)) {
      csvData = fs.readFileSync(dataPath, 'utf8');
    } else {
      // Try to find the file in the project structure
      const possiblePaths = [
        path.join(process.cwd(), '../news/nvidia_score_price_dump.txt'),
        path.join(process.cwd(), '../../news/nvidia_score_price_dump.txt'),
        path.join(process.cwd(), 'nvidia_score_price_dump.txt')
      ];
      
      for (const possiblePath of possiblePaths) {
        if (fs.existsSync(possiblePath)) {
          csvData = fs.readFileSync(possiblePath, 'utf8');
          break;
        }
      }
      
      if (!csvData) {
        throw new Error('Data file not found');
      }
    }

    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.status(200).send(csvData);
    
  } catch (error) {
    console.error('Error reading data file:', error);
    
    // Fallback to reading from the local copy
    try {
      const localData = fs.readFileSync('./nvidia_score_price_dump.txt', 'utf8');
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Content-Type', 'text/plain');
      res.status(200).send(localData);
    } catch (fallbackError) {
      res.status(500).json({ error: 'Unable to load data file' });
    }
  }
}