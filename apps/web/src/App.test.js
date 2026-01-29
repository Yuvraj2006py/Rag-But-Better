import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import App from "./App.jsx";

describe("App", () => {
  it("renders title", () => {
    const { getByText } = render(<App />);
    expect(getByText("RAG but Better")).toBeTruthy();
  });
});
