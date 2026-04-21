export interface LogContext {
  module?: string;
  function?: string;
  operation?: string;
  [key: string]: any;
}

export class AppLogger {
  private context: LogContext;
  private logLevel: string;
  private useColors: boolean;

  constructor(context: LogContext = {}) {
    this.context = context;
    this.logLevel = process.env.LOG_LEVEL || 'info';
    this.useColors = process.env.NO_COLOR !== '1' && process.stdout.isTTY;
  }

  private getColor(level: string): string {
    if (!this.useColors) return '';
    
    const colors = {
      debug: '\x1b[36m',     // Cyan
      info: '\x1b[32m',      // Green
      warn: '\x1b[33m',      // Yellow
      error: '\x1b[31m',     // Red
      reset: '\x1b[0m',      // Reset
      dim: '\x1b[2m',        // Dim
      bright: '\x1b[1m'      // Bright
    };
    
    return colors[level as keyof typeof colors] || '';
  }

  private getCallerInfo(): string {
    const stack = new Error().stack;
    if (!stack) return 'unknown:0';
    
    const lines = stack.split('\n');
    // Skip the current line and the logger method line
    for (let i = 3; i < lines.length; i++) {
      const line = lines[i];
      // Match TypeScript/JavaScript file patterns
      const match = line.match(/at.*\((.*\.ts|.*\.js):(\d+):(\d+)\)/) || 
                   line.match(/at.*(.*\.ts|.*\.js):(\d+):(\d+)/) ||
                   line.match(/at.*(.*\.ts|.*\.js):(\d+)/);
      
      if (match) {
        const filePath = match[1];
        const lineNumber = match[2];
        // Extract just the filename from the path
        const fileName = filePath.split('/').pop() || filePath.split('\\').pop() || filePath;
        const cleanFileName = fileName.split(':').shift() || fileName; // Remove any remaining path parts
        return `${cleanFileName}:${lineNumber}`;
      }
    }
    
    return 'unknown:0';
  }

  private formatMessage(message: string, level: string): string {
    const timestamp = new Date().toISOString();
    const callerInfo = this.getCallerInfo();
    return `[${timestamp}] ${level.toUpperCase()} ${callerInfo} ${message}`;
  }

  private shouldLog(level: string): boolean {
    const levels = ['debug', 'info', 'warn', 'error'];
    const currentLevelIndex = levels.indexOf(this.logLevel);
    const messageLevelIndex = levels.indexOf(level);
    return messageLevelIndex >= currentLevelIndex;
  }

  private colorize(message: string, level: string): string {
    if (!this.useColors) return message;
    
    const color = this.getColor(level);
    const reset = this.getColor('reset');
    const dim = this.getColor('dim');
    const bright = this.getColor('bright');
    
    // Colorize different parts of the message
    const timestampMatch = message.match(/^(\[.*?\])/);
    const levelMatch = message.match(/\[.*?\]\s+(\w+)/);
    const fileMatch = message.match(/\[.*?\]\s+\w+\s+([^:]+:\d+)/);
    
    if (timestampMatch && levelMatch && fileMatch) {
      const timestamp = timestampMatch[1];
      const level = levelMatch[1];
      const file = fileMatch[1];
      const rest = message.slice(fileMatch.index! + fileMatch[0].length + 1);
      
      return `${dim}${timestamp}${reset} ${color}${level}${reset} ${dim}${file}${reset} ${rest}`;
    }
    
    return `${color}${message}${reset}`;
  }

  debug(message: string, context?: LogContext): void {
    if (this.shouldLog('debug')) {
      const formattedMessage = this.formatMessage(message, 'debug');
      console.debug(this.colorize(formattedMessage, 'debug'));
    }
  }

  info(message: string, context?: LogContext): void {
    if (this.shouldLog('info')) {
      const formattedMessage = this.formatMessage(message, 'info');
      console.info(this.colorize(formattedMessage, 'info'));
    }
  }

  warn(message: string, context?: LogContext): void {
    if (this.shouldLog('warn')) {
      const formattedMessage = this.formatMessage(message, 'warn');
      console.warn(this.colorize(formattedMessage, 'warn'));
    }
  }

  error(message: string, error?: Error | unknown, context?: LogContext): void {
    const errorMessage = error ? 
      `${message} (Error: ${error instanceof Error ? error.message : String(error)})` : 
      message;
    
    if (this.shouldLog('error')) {
      const formattedMessage = this.formatMessage(errorMessage, 'error');
      console.error(this.colorize(formattedMessage, 'error'));
    }
  }

  child(context: LogContext): AppLogger {
    return new AppLogger({ ...this.context, ...context });
  }
}

export const logger = new AppLogger({ module: 'rag-retrieval-core' });
