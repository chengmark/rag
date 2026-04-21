import { Logger, ILogObj } from 'tslog';

export const logger = new Logger<ILogObj>({
  name: 'rag-indexing',
  type: 'pretty',
  stylePrettyLogs: true,
  minLevel: (process.env.LOG_LEVEL || 'info') as any,
  prettyInspectOptions: {
    depth: 3,
    colors: process.env.NO_COLOR !== '1'
  }
});
