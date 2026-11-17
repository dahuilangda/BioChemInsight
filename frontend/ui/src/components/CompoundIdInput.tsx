import React from 'react';

interface Props {
  initialValue: string;
  rowIndex: number;
  disabled?: boolean;
  onSave: (rowIndex: number, value: string) => void;
}

const CompoundIdInput = React.memo(function CompoundIdInput({ initialValue, rowIndex, disabled, onSave }: Props) {
  const [value, setValue] = React.useState(initialValue);

  React.useEffect(() => {
    setValue(initialValue);
  }, [initialValue]);

  const canceled = React.useRef(false);
  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      event.currentTarget.blur();
      event.stopPropagation();
    } else if (event.key === 'Escape') {
      event.stopPropagation(); canceled.current = true; setValue(initialValue); event.currentTarget.blur();
    }
  };

  const handleBlur = () => {
    if (!canceled.current && value !== initialValue) onSave(rowIndex, value);
  };

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setValue(event.target.value);
  };

  return (
    <input
      type="text"
      className="page-cell__id-input"
      value={value}
      placeholder="Compound ID" aria-label={`Compound ID for row ${rowIndex + 1}`}
      onKeyDown={handleKeyDown}
      onFocus={() => {canceled.current = false;}}
      onBlur={handleBlur}
      onChange={handleChange}
      disabled={disabled}
    />
  );
});

export default CompoundIdInput;
