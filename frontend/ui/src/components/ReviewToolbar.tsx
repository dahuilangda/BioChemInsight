import React from 'react';
import ReviewPopover from './ReviewPopover';
import {
  activeFilter,
  type ReviewQuery,
  type FilterRule,
  type OrderRule,
} from '../utils/reviewQuery';
export type ReviewMenu =
  | 'columns'
  | 'filter'
  | 'group'
  | 'sort'
  | 'display'
  | 'search'
  | 'add'
  | 'export'
  | 'more';
export interface ReviewToolbarHandle {
  open: (menu: ReviewMenu) => void;
  anchor: (menu: ReviewMenu | 'history') => HTMLButtonElement | null;
}
function Glyph({ kind }: { kind: string }) {
  const paths: Record<string, React.ReactNode> = {
    columns: (
      <>
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <path d="M9 4v16M15 4v16" />
      </>
    ),
    filter: <path d="M3 5h18M6 12h12M10 19h4" />,
    sort: (
      <>
        <path d="M7 4v16m-3-3 3 3 3-3M14 6h7M14 12h5M14 18h3" />
      </>
    ),
    group: (
      <>
        <rect x="3" y="4" width="18" height="5" rx="1" />
        <rect x="3" y="14" width="18" height="6" rx="1" />
      </>
    ),
    search: (
      <>
        <circle cx="10" cy="10" r="6" />
        <path d="m15 15 6 6" />
      </>
    ),
    history: (
      <>
        <path d="M3 11a9 9 0 1 1 2 7M3 4v7h7M12 7v5l3 2" />
      </>
    ),
    add: <path d="M12 4v16M4 12h16" />,
    export: <path d="M12 3v12m-4-4 4 4 4-4M4 17v4h16v-4" />,
    more: (
      <>
        <circle cx="5" cy="12" r="1" />
        <circle cx="12" cy="12" r="1" />
        <circle cx="19" cy="12" r="1" />
      </>
    ),
    display: (
      <>
        <path d="M3 6h18M3 12h18M3 18h18" />
        <circle cx="8" cy="6" r="2" />
        <circle cx="16" cy="12" r="2" />
        <circle cx="10" cy="18" r="2" />
      </>
    ),
  };
  return (
    <svg
      className="button-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      aria-hidden="true"
    >
      {paths[kind]}
    </svg>
  );
}
interface Props {
  columns: React.ReactNode;
  display: React.ReactNode;
  view: React.ReactNode;
  shortcuts: React.ReactNode;
  pager: React.ReactNode;
  status: React.ReactNode;
  focus: React.ReactNode;
  query: ReviewQuery;
  onQuery: (query: ReviewQuery) => void;
  fields: { value: string; label: string }[];
  search: string;
  onSearch: (value: string) => void;
  onHistory: () => void;
  historyOpen: boolean;
  canEdit: boolean;
  onAddCompound: () => void;
  onAddMeasurement: () => void;
  onReset: () => void;
  onUndo: () => void;
  onReview: () => void;
  onRecover: () => void;
  onExport: () => void;
  exportOptions?: React.ReactNode;
  exporting: boolean;
  onCollapseGroups: () => void;
  onExpandGroups: () => void;
}
const names: Record<ReviewMenu, string> = {
  columns: 'Columns',
  filter: 'Filter',
  group: 'Group',
  sort: 'Sort',
  display: 'View options',
  search: 'Search results',
  add: 'Add data',
  more: 'More actions',
  export: 'Export',
};
const ReviewToolbar = React.forwardRef<ReviewToolbarHandle, Props>(
  function ReviewToolbar(props, ref) {
    const [menu, setMenu] = React.useState<ReviewMenu | null>(null);
    const anchors = React.useRef<Partial<Record<ReviewMenu, HTMLButtonElement | null>>>({});
    const history = React.useRef<HTMLButtonElement>(null);
    const openMenu = (key: ReviewMenu) => {
      if (props.historyOpen) props.onHistory();
      if (key === 'filter' && !props.query.filters.length)
        props.onQuery({ ...props.query, filters: [{ column: '', operator: 'filled', value: '' }] });
      if (key === 'sort' && !props.query.sorts.length)
        props.onQuery({ ...props.query, sorts: [{ column: '', descending: false }] });
      if (key === 'group' && !props.query.groups.length)
        props.onQuery({ ...props.query, groups: [{ column: '', descending: false }] });
      setMenu(key);
    };
    React.useImperativeHandle(ref, () => ({
      open: openMenu,
      anchor: (menu: ReviewMenu | 'history') =>
        menu === 'history' ? history.current : anchors.current[menu] || null,
    }));
    const toggle = (key: ReviewMenu) => (menu === key ? setMenu(null) : openMenu(key));
    const updateFilter = (index: number, patch: Partial<FilterRule>) =>
      props.onQuery({
        ...props.query,
        filters: props.query.filters.map((r, i) => (i === index ? { ...r, ...patch } : r)),
      });
    const updateOrder = (key: 'sorts' | 'groups', index: number, patch: Partial<OrderRule>) =>
      props.onQuery({
        ...props.query,
        [key]: props.query[key].map((r, i) => (i === index ? { ...r, ...patch } : r)),
      });
    const fieldSelect = (value: string, onChange: (value: string) => void, label: string) => (
      <select
        id={label === 'Sort column' ? 'review-sort-column' : undefined}
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="">Choose field</option>
        {props.fields.map((field) => (
          <option key={field.value} value={field.value}>
            {field.label}
          </option>
        ))}
      </select>
    );
    const command = (key: ReviewMenu, count = 0, iconOnly = false) => (
      <button
        ref={(node) => {
          anchors.current[key] = node;
        }}
        type="button"
        data-review-command={key}
        className={`review-command ${key === 'columns' ? 'columns-picker__toggle' : ''} ${key === 'filter' ? 'review-filters-toggle' : ''} ${key === 'search' ? 'review-search-toggle' : ''} ${key === 'add' ? 'review-manage-menu' : ''} ${menu === key ? 'is-open' : ''} ${count ? 'has-filter' : ''}`}
        aria-label={key === 'filter' ? 'Filters' : names[key]}
        aria-expanded={menu === key}
        title={names[key]}
        onClick={() => toggle(key)}
      >
        <Glyph kind={key} />
        {!iconOnly && <span>{key === 'add' ? 'Add' : names[key]}</span>}
        {count > 0 && <small>{count}</small>}
      </button>
    );
    const run = (action: () => void) => () => {
      setMenu(null);
      action();
    };
    const orders = (key: 'sorts' | 'groups') => (
      <div className="review-rule-list">
        {props.query[key].map((rule, index) => (
          <div className="review-order-rule" key={index}>
            {fieldSelect(
              rule.column,
              (value) => updateOrder(key, index, { column: value }),
              key === 'sorts'
                ? index === 0
                  ? 'Sort column'
                  : `Sort column ${index + 1}`
                : `Group field ${index + 1}`,
            )}
            <select
              aria-label={
                key === 'sorts'
                  ? index === 0
                    ? 'Sort direction'
                    : `Sort direction ${index + 1}`
                  : `Group direction ${index + 1}`
              }
              value={rule.descending ? 'desc' : 'asc'}
              onChange={(e) => updateOrder(key, index, { descending: e.target.value === 'desc' })}
            >
              <option value="asc">Ascending</option>
              <option value="desc">Descending</option>
            </select>
            <span className="review-rule-actions">
              {([-1, 1] as const).map((direction) => (
                <button
                  type="button"
                  className="review-icon-button"
                  key={direction}
                  disabled={index + direction < 0 || index + direction >= props.query[key].length}
                  aria-label={`Move ${key === 'sorts' ? 'sort' : 'group'} ${index + 1} ${direction < 0 ? 'up' : 'down'}`}
                  onClick={() => {
                    const rules = [...props.query[key]];
                    [rules[index], rules[index + direction]] = [
                      rules[index + direction],
                      rules[index],
                    ];
                    props.onQuery({ ...props.query, [key]: rules });
                  }}
                >
                  {direction < 0 ? '↑' : '↓'}
                </button>
              ))}
              <button
                className="review-icon-button"
                aria-label={`Remove ${key === 'sorts' ? 'sort' : 'group'} ${index + 1}`}
                onClick={() =>
                  props.onQuery({
                    ...props.query,
                    [key]: props.query[key].filter((_, i) => i !== index),
                  })
                }
              >
                ×
              </button>
            </span>
          </div>
        ))}
        <button
          className="review-text-button"
          disabled={key === 'groups' && props.query.groups.length >= 3}
          onClick={() =>
            props.onQuery({
              ...props.query,
              [key]: [
                ...props.query[key],
                {
                  column:
                    props.fields.find((f) => !props.query[key].some((r) => r.column === f.value))
                      ?.value || '',
                  descending: false,
                },
              ],
            })
          }
        >
          + {key === 'sorts' ? 'Add sort' : 'Add group'}
        </button>
      </div>
    );
    return (
      <div className="review-toolbar review-toolbar--unified">
        <div className="jobs-toolbar review-toolbar__controls">
          {props.view}
          {command('columns')}
          {command('filter', props.query.filters.filter(activeFilter).length)}
          {command('group', props.query.groups.filter((r) => r.column).length)}
          {command('sort', props.query.sorts.filter((r) => r.column).length)}
          {command('search', props.search ? 1 : 0, true)}
          <span className="review-toolbar-divider" />
          <button
            ref={history}
            type="button"
            data-review-command="history"
            className={`review-command ${props.historyOpen ? 'is-open' : ''}`}
            disabled={!props.canEdit}
            aria-expanded={props.historyOpen}
            onClick={() => {
              setMenu(null);
              props.onHistory();
            }}
          >
            <Glyph kind="history" />
            <span>History</span>
          </button>
          {command('add')}
          {props.canEdit &&
            (props.exportOptions ? (
              command('export')
            ) : (
              <button
                className="review-command"
                type="button"
                aria-label="Export"
                title="Export results"
                disabled={!props.canEdit || props.exporting}
                onClick={props.onExport}
              >
                <Glyph kind="export" />
                <span>Export</span>
              </button>
            ))}
          {command('display', 0, true)}
          {command('more', 0, true)}
          {props.focus}
          {props.shortcuts}
          {props.status}
          {props.pager}
        </div>
        {menu && (
          <ReviewPopover
            anchor={anchors.current[menu] || null}
            title={names[menu]}
            onClose={() => setMenu(null)}
            width={menu === 'filter' ? 460 : menu === 'sort' || menu === 'group' ? 420 : 320}
            className={`review-command-popover review-command-popover--${menu}`}
          >
            {menu === 'columns' && <div className="columns-picker__panel">{props.columns}</div>}
            {menu === 'search' && (
              <div id="review-search-panel" className="review-search-content">
                <input
                  autoFocus
                  type="search"
                  aria-label="Search results"
                  placeholder="Find compound or measurement…"
                  value={props.search}
                  onChange={(e) => props.onSearch(e.target.value)}
                />
                {props.search && (
                  <button className="review-text-button" onClick={() => props.onSearch('')}>
                    Clear search
                  </button>
                )}
              </div>
            )}
            {menu === 'filter' && (
              <div id="review-advanced-filters">
                <p className="review-menu-note">
                  Show records matching{' '}
                  <select
                    aria-label="Match filter conditions"
                    value={props.query.match}
                    onChange={(e) =>
                      props.onQuery({ ...props.query, match: e.target.value as 'all' | 'any' })
                    }
                  >
                    <option value="all">all conditions</option>
                    <option value="any">any condition</option>
                  </select>
                </p>
                <div className="review-rule-list">
                  {props.query.filters.map((rule, index) => (
                    <div className="review-filter-rule" key={index}>
                      {fieldSelect(
                        rule.column,
                        (value) => updateFilter(index, { column: value }),
                        index === 0 ? 'Filter column' : `Filter column ${index + 1}`,
                      )}
                      <select
                        aria-label={
                          index === 0 ? 'Filter condition' : `Filter condition ${index + 1}`
                        }
                        value={rule.operator}
                        onChange={(e) => updateFilter(index, { operator: e.target.value })}
                      >
                        {[
                          ['filled', 'Is not empty'],
                          ['empty', 'Is empty'],
                          ['equals', 'Equals'],
                          ['not_equals', 'Does not equal'],
                          ['contains', 'Contains'],
                          ['not_contains', 'Does not contain'],
                          ['gt', 'Greater than'],
                          ['gte', 'At least'],
                          ['lt', 'Less than'],
                          ['lte', 'At most'],
                        ].map(([value, label]) => (
                          <option key={value} value={value}>
                            {label}
                          </option>
                        ))}
                      </select>
                      {!['filled', 'empty'].includes(rule.operator) && (
                        <input
                          aria-label={index === 0 ? 'Filter value' : `Filter value ${index + 1}`}
                          value={rule.value}
                          placeholder="Value"
                          onChange={(e) => updateFilter(index, { value: e.target.value })}
                        />
                      )}
                      <button
                        className="review-icon-button"
                        aria-label={`Remove condition ${index + 1}`}
                        onClick={() =>
                          props.onQuery({
                            ...props.query,
                            filters: props.query.filters.filter((_, i) => i !== index),
                          })
                        }
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
                <div className="review-menu-footer">
                  <button
                    className="review-text-button"
                    onClick={() =>
                      props.onQuery({
                        ...props.query,
                        filters: [
                          ...props.query.filters,
                          { column: '', operator: 'filled', value: '' },
                        ],
                      })
                    }
                  >
                    + Add condition
                  </button>
                  {props.query.filters.length > 0 && (
                    <button
                      className="review-text-button"
                      onClick={() => props.onQuery({ ...props.query, filters: [] })}
                    >
                      Clear filters
                    </button>
                  )}
                </div>
              </div>
            )}
            {menu === 'sort' && (
              <>
                <p className="review-menu-note">
                  Apply rules from top to bottom. Empty values stay last.
                </p>
                {orders('sorts')}
                {props.query.sorts.length > 0 && (
                  <button
                    className="review-text-button"
                    onClick={() => props.onQuery({ ...props.query, sorts: [] })}
                  >
                    Clear sorts
                  </button>
                )}
              </>
            )}
            {menu === 'group' && (
              <>
                <p className="review-menu-note">Group matching values. Records remain editable.</p>
                {orders('groups')}
                <div className="review-menu-footer">
                  <button
                    className="review-text-button"
                    disabled={!props.query.groups.length}
                    onClick={props.onCollapseGroups}
                  >
                    Collapse all
                  </button>
                  <button
                    className="review-text-button"
                    disabled={!props.query.groups.length}
                    onClick={props.onExpandGroups}
                  >
                    Expand all
                  </button>
                  {props.query.groups.length > 0 && (
                    <button
                      className="review-text-button"
                      onClick={() => props.onQuery({ ...props.query, groups: [] })}
                    >
                      Clear groups
                    </button>
                  )}
                </div>
              </>
            )}
            {menu === 'display' && (
              <div className="review-display-options">
                {props.display}
                <button className="review-text-button" onClick={run(props.onReset)}>
                  Reset view
                </button>
              </div>
            )}
            {menu === 'add' && (
              <div className="review-action-list">
                <button disabled={!props.canEdit} onClick={run(props.onAddCompound)}>
                  Add compound
                </button>
                <button disabled={!props.canEdit} onClick={run(props.onAddMeasurement)}>
                  Add measurement
                </button>
              </div>
            )}
            {menu === 'more' && (
              <div className="review-action-list">
                <button disabled={!props.canEdit} onClick={run(props.onUndo)}>
                  Undo last save
                </button>
                <button onClick={run(props.onReview)}>Needs review</button>
                <button disabled={!props.canEdit} onClick={run(props.onRecover)}>
                  Find missing data
                </button>
              </div>
            )}
            {menu === 'export' && (
              <div className="review-action-list" onClick={() => setMenu(null)}>
                {props.exportOptions}
              </div>
            )}
          </ReviewPopover>
        )}
      </div>
    );
  },
);
export default ReviewToolbar;
