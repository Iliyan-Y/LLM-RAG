import { v4 as uuidv4 } from "uuid";
import styles from "./Answer.module.css";

const Answer = ({ text }: { text: string }) => {
	const lines = text.split("\n");
	return (
		<div className={styles.container}>
			{lines.map((line) => (
				<p
					key={uuidv4()}
					className={`${line[0] === "-" ? styles.list : styles.line}`}
				>
					{line}
				</p>
			))}
		</div>
	);
};

export default Answer;
