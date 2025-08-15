import { useEffect, useState } from "react";
import type { Message } from "../App";
import { v4 as uuidv4 } from "uuid";
import styles from "./Message.module.css";

type LlmResponse = {
	answer: string;
	sources: Source[];
};

interface Source {
	source: string;
	filing_type: string;
	period_end_date: string;
	page_label: string;
}

const MessageComponent = ({
	message,
	isLast,
}: {
	message: Message;
	isLast: boolean;
}) => {
	const [llmResponse, setLlmResponse] = useState<LlmResponse | null>(null);
	const [showSources, setShowSources] = useState(false);
	useEffect(() => {
		try {
			const parsed = JSON.parse(message.text);
			if (parsed.answer) {
				setLlmResponse(parsed);
			} else {
				setLlmResponse({ answer: parsed.message, sources: [] });
			}
		} catch {
			setLlmResponse({ answer: message.text, sources: [] });
		}
	}, []);

	return (
		<div
			key={message.id}
			className={`message ${message.sender === "me" ? "me" : "other"}`}
		>
			<div
				className={`message-text ${
					llmResponse && llmResponse.answer === "processing" && isLast
						? styles.processing
						: ""
				}`}
			>
				{llmResponse ? llmResponse.answer : message.text}
			</div>
			{llmResponse && llmResponse.sources.length > 0 && (
				<div>
					<p
						onClick={() => setShowSources((prev) => !prev)}
						style={{ cursor: "pointer", color: "hotpink" }}
					>
						showing sources
					</p>
					{showSources &&
						llmResponse.sources.map((source) => (
							<div key={uuidv4()}>
								<p>
									Source: {source.source}, Page: {source.page_label}, Date:
									{source.period_end_date}, Type: {source.filing_type}
								</p>
							</div>
						))}
				</div>
			)}
			<div className="message-meta">
				{new Date(message.time).toLocaleTimeString()}
			</div>
		</div>
	);
};

export default MessageComponent;
