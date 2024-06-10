import { Dataset } from "./dataset.js";
import { Image } from "./image.js"

export { Image };
export type Tabular = Partial<Record<string, string>>;
export type Text = string;

// TODO get rid of it when fully typed
export type TypedDataset =
  | ["image", Dataset<[Image, label: string]>]
  | ["tabular", Dataset<Tabular>]
  | ["text", Dataset<Text>];
