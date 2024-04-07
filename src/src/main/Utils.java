package main;
import com.alibaba.fastjson.JSONArray;
import org.apache.calcite.config.Lex;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptUtil;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import org.apache.calcite.prepare.RelOptTableImpl;
import org.apache.calcite.rel.RelFieldCollation;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.*;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.Planner;
import org.apache.calcite.util.SourceStringReader;

import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.apache.calcite.avatica.util.Casing.UNCHANGED;
import static org.apache.calcite.avatica.util.Quoting.DOUBLE_QUOTE;

public class Utils {

    public static void writeContentStringToLocalFile(String str,String path){
        try {
            File file = new File(path);
            if (!file.getParentFile().exists()) {
                file.getParentFile().mkdirs();
            }
            file.createNewFile();
            if(str != null && !"".equals(str)){
                FileWriter fw = new FileWriter(file, true);
                fw.write(str);
                fw.flush();
                fw.close();
                System.out.println("执行完毕!");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static String readStringFromFile(String filePath) {
        String contentStr = "";
        try {
            File jsonFile = new File(filePath);
            FileReader fileReader = new FileReader(jsonFile);
            Reader reader = new InputStreamReader(new FileInputStream(jsonFile),"utf-8");
            int ch = 0;
            StringBuffer sb = new StringBuffer();
            while ((ch = reader.read()) != -1) {
                sb.append((char) ch);
            }
            fileReader.close();
            reader.close();
            contentStr = sb.toString();
            return contentStr;
        } catch (IOException e) {
            e.printStackTrace();
            return "[]";
        }
    }

    public static String[] readWorkloadFromFile(String filePath) {
        String contentStr = Utils.readStringFromFile(filePath);
        String[] result = contentStr.split(";");
        return result;
    }

    public static JSONArray readJsonFile(String filePath) {
        String contentStr = Utils.readStringFromFile(filePath);
        return JSON.parseArray(contentStr);
    }
    public static JSONObject generate_json(Node node) throws Exception {
        JSONObject res = new JSONObject();
        res.put("name",node.name);
        res.put("plan", RelOptUtil.toString(node.state_rel));
        res.put("cost", node.rewriter.getCostRecordFromRelNode(node.state_rel));
        Map tmp = new HashMap();
        for(Object k : node.activatedRules.keySet()){
            tmp.put(((RelOptRule) k).toString(),node.activatedRules.get(k));
        }
        int tsz = 0;
        for(Object k :node.rewrite_sequence){
            res.put(String.format("act_rule_%d",++tsz),k);
        }
        res.put("activated_rules",tmp);

        List children = node.children;
        List<JSONObject> children_jsons = new ArrayList<>();
        for(int i = 0;i<children.size();i++){
            children_jsons.add(generate_json((Node) children.get(i)));
        }
        res.put("children",children_jsons);
        return res;
    }

    public static JSONObject getPredicate(RexCall predicates) {
        Collection<SqlKind> collection = new ArrayList();
        collection.add(SqlKind.PLUS);
        collection.add(SqlKind.MINUS);
        collection.add(SqlKind.TIMES);
        collection.add(SqlKind.DIVIDE);

        System.out.println("predicates:" + predicates);
        RexNode columnRexNode = null;
        RexNode valueRexNode = null;
        if(!(predicates.getOperands().get(1) instanceof RexInputRef)){
            columnRexNode = predicates.getOperands().get(0);
            valueRexNode = predicates.getOperands().get(1);
        }else if(!(predicates.getOperands().get(0) instanceof RexInputRef))  {
            columnRexNode = predicates.getOperands().get(1);
            valueRexNode = predicates.getOperands().get(0);
        }else {
            System.out.println("意外之喜predicates:" + predicates);
            return null;
        }
        System.out.println("columnRexNode:" + columnRexNode);
        System.out.println("valueRexNode:" + valueRexNode);

        if (columnRexNode.isA(collection)) {
            // 过滤列的加减乘除
            System.out.println("列的加减乘除不能处理");
            return null;
        }
        String reg = "\\$\\d+";
        Pattern p = Pattern.compile(reg);
        Matcher m = p.matcher(columnRexNode.toString());
        int columnIndex = 0;
        if(m.find()){
            columnIndex = Integer.parseInt(m.group().replace("$", ""));
        }else {
            // 未发现列的索引
            return null;
        }
        String valueString = valueRexNode.toString();
        //  数值的加减乘除计算
        if (valueRexNode.isA(collection)) {
            valueString = Utils.calculateRexNodeString(valueString);
        }else {
            // 去掉DECIMAL标识
            valueString = Utils.formatRexNodeString(valueString);
        }
        if (valueString.contains("$")) {
            System.out.println("Value值中有列:" + valueString);
            return null;
        }
        String oper = predicates.getOperator().toString();
        JSONObject predicate_json = new JSONObject();
        predicate_json.put("column_index", columnIndex);
        predicate_json.put("const",valueString);
        predicate_json.put("operator",oper);
        System.out.println("Success:" + predicate_json);
        return predicate_json;
    }

    public static JSONArray process_clause(RexCall predicates, String logic_oper1, RelNode node){
        Utils.getPredicate(predicates);
        try {
            String oper = predicates.getOperator().toString();
            if (oper == "AND" || oper == "OR"){
                JSONArray resList = new JSONArray();
                if (predicates.getOperands().get(0) instanceof RexCall) {
                    RexCall subpred1 = (RexCall) predicates.getOperands().get(0);
                    JSONArray res_l = process_clause(subpred1,logic_oper1,node);
                    resList.addAll(res_l);
                    for (int i = 1; i < predicates.getOperands().size(); i++) {
                        String logic_oper = oper;
                        RexCall subpred = (RexCall) predicates.getOperands().get(i);
                        JSONArray res = process_clause(subpred,logic_oper,node);
                        resList.addAll(res);
                    }
                }
//                System.out.println("process_clause if 结束:" + resList.toJSONString());
                return resList;
            }else {
                Collection<SqlKind> collection = new ArrayList();
                collection.add(SqlKind.PLUS);
                collection.add(SqlKind.MINUS);
                collection.add(SqlKind.TIMES);
                collection.add(SqlKind.DIVIDE);
//                System.out.println("process_clause else 开始:" + predicates);
                JSONArray resList = new JSONArray();
                if (predicates.getOperands().size() == 0) {
//                    System.out.println("process_clause else 结束:" + resList.toJSONString());
                    return resList;
                }else if (predicates.getOperands().size() == 1) {
                    if(!(predicates.getOperands().get(0) instanceof RexInputRef)){
                        String logic_oper = oper;
                        RexCall subpred = (RexCall) predicates.getOperands().get(0);
                        JSONArray res = process_clause(subpred,logic_oper,node);
                        resList.addAll(res);
//                        System.out.println("process_clause else 结束:" + resList.toJSONString());
                        return resList;
                    }
                } else {
                    if(!(predicates.getOperands().get(1) instanceof RexInputRef)){
                        String  operand1 =  predicates.getOperands().get(0).toString();
                        // 过滤列的加减乘除
                        if (predicates.getOperands().get(0).isA(collection)) {
                            return resList;
                        }
                        String reg = "\\$\\d+";
                        Pattern p = Pattern.compile(reg);
                        Matcher m = p.matcher(operand1);
                        int column_index = 0;
                        if(m.find()){
                            column_index = Integer.parseInt(m.group().replace("$", ""));
                        }else {
                            return resList;
                        }
                        String tableNcolumn = get_relative_columns_with_type(node.getInputs(),column_index);
                        String column_name = tableNcolumn.split("\\.")[1];
                        String  operand2 =  predicates.getOperands().get(1).toString();

                        //  数值的加减乘除计算
                        if (predicates.getOperands().get(1).isA(collection)) {
                            operand2 = Utils.calculateRexNodeString(operand2);
                        }else {
                            // 去掉DECIMAL标识
                            operand2 = Utils.formatRexNodeString(operand2);
                        }

                        JSONObject predicate_json = new JSONObject();
                        predicate_json.put("column",column_name);
                        predicate_json.put("const",operand2);
                        predicate_json.put("context",logic_oper1);
                        predicate_json.put("operator",oper);
                        resList.add(predicate_json);
                        return resList;
                    }
                    else if(!(predicates.getOperands().get(0) instanceof RexInputRef)){
                        String  operand1 =  predicates.getOperands().get(1).toString();

                        // 过滤列的加减乘除
                        if (predicates.getOperands().get(1).isA(collection)) {
                            return resList;
                        }

                        String reg = "\\$\\d+";
                        Pattern p = Pattern.compile(reg);
                        Matcher m = p.matcher(operand1);
                        int column_index = 0;
                        if(m.find()){
                            column_index = Integer.parseInt(m.group().replace("$", ""));
                        }else {
                            return resList;
                        }
                        String tableNcolumn = get_relative_columns_with_type(node.getInputs(),column_index);
                        String column_name = tableNcolumn.split("\\.")[1];
                        String  operand2 =  predicates.getOperands().get(0).toString();
                        // 数值的加减乘除计算
                        if (predicates.getOperands().get(0).isA(collection)) {
                            operand2 = Utils.calculateRexNodeString(operand2);
                        }else {
                            // 去掉DECIMAL标识
                            operand2 = Utils.formatRexNodeString(operand2);
                        }
                        JSONObject predicate_json = new JSONObject();
                        predicate_json.put("column",column_name);
                        predicate_json.put("const",operand2);
                        predicate_json.put("context",logic_oper1);
                        predicate_json.put("operator",oper);
                        resList.add(predicate_json);
                        return resList;
                    }
                }
            }
        } catch (Exception error) {
            error.printStackTrace();
            System.out.println("error:" + error);
        }
        return new JSONArray();
    }

    public static String formatRexNodeString(String rexNodeValue) {
        rexNodeValue = rexNodeValue.replaceAll(" ", "");
        String removeDecimalReg = "[:DECIMAL]+?[/(]\\d+,\\d+[/)]";
        Pattern removeDecimalPattern = Pattern.compile(removeDecimalReg);
        Matcher removeDecimalMatcher = removeDecimalPattern.matcher(rexNodeValue);
        rexNodeValue =removeDecimalMatcher.replaceAll("");
        return rexNodeValue;
    }

    public static String calculateRexNodeString(String rexNodeValue) {
        System.out.println("calculateRexNodeString:"+rexNodeValue);
        try {

            rexNodeValue = Utils.formatRexNodeString(rexNodeValue);

            String reg1 = "[\\*|\\+|\\-|\\/]\\([\\-|\\+]?\\d+(\\.\\d+)*,[\\-|\\+]?\\d+(\\.\\d+)*?\\)";
            Pattern p1 = Pattern.compile(reg1);
            Matcher m1 = p1.matcher(rexNodeValue);
            if(m1.find()){
                String group = m1.group();
                String reg2 = "(\\-|\\+)?\\d+(\\.\\d+)?";
                Pattern p2 = Pattern.compile(reg2);
                Matcher m2 = p2.matcher(group);
                ArrayList resultList = new ArrayList<>();
                if (m2.groupCount() == 2) {
                    while(m2.find()){
                        resultList.add(m2.group(0));
                    }
                }
                String operator = group.substring(0, 1);
                double operatorResult = 0;
                if (operator.equalsIgnoreCase("+")) {
                    operatorResult = Double.valueOf((String)resultList.get(0)) + Double.valueOf((String)resultList.get(1));
                }else if (operator.equalsIgnoreCase("-")) {
                    operatorResult = Double.valueOf((String)resultList.get(0)) - Double.valueOf((String)resultList.get(1));
                }else if (operator.equalsIgnoreCase("*")) {
                    operatorResult = Double.valueOf((String)resultList.get(0)) * Double.valueOf((String)resultList.get(1));
                }else if (operator.equalsIgnoreCase("/")) {
                    operatorResult = Double.valueOf((String)resultList.get(0)) / Double.valueOf((String)resultList.get(1));
                }
                rexNodeValue = m1.replaceFirst(String.valueOf(operatorResult));
                return Utils.calculateRexNodeString(rexNodeValue);
            }
        }catch (Exception e) {
            e.printStackTrace();
        }
        return rexNodeValue;
    }

    public static JSONObject getConditionFromRelNode(RelNode rel_node) {

        JSONObject parse_res = new JSONObject();
        try {
            Deque stack = new LinkedList();
            stack.add(rel_node);
            JSONArray res = new JSONArray();
            while (stack.size()>0) {
                RelNode node = (RelNode) stack.pop();
                for (int i = 0; i < node.getInputs().size(); i++) {
                    stack.add(node.getInputs().get(i));
                }
                JSONArray clause_res;
                RexNode condition = null;
                if (node instanceof LogicalFilter) {
                    condition = ((LogicalFilter) node).getCondition();
                } else if (node instanceof LogicalJoin) {
                    condition = ((LogicalJoin) node).getCondition();
                }
                if (condition instanceof RexCall) {
                    clause_res = process_clause((RexCall) condition, "null" ,node);
                    res.addAll(clause_res);
                }
            }
            parse_res.put("conditions",res);
        } catch (Exception e) {
            e.printStackTrace();
            parse_res.put("conditions", new ArrayList());
        }
        return parse_res;
    }

    static String get_relative_columns_with_type (List<RelNode> childs, int index){
        String res = "";
        if (index<childs.get(0).getRowType().getFieldNames().size()){
            if(childs.get(0).getCluster().getMetadataQuery().getColumnOrigin(childs.get(0),index) != null){
                res += childs.get(0).getCluster().getMetadataQuery().getColumnOrigin(childs.get(0),index).getOriginTable().getQualifiedName().get(0);
                res += "."+childs.get(0).getCluster().getMetadataQuery().getColumnOrigin(childs.get(0),index).getOriginTable().getRowType().getFieldNames().get(childs.get(0).getCluster().getMetadataQuery().getColumnOrigin(childs.get(0),index).getOriginColumnOrdinal());
                res += "."+childs.get(0).getCluster().getMetadataQuery().getColumnOrigin(childs.get(0),index).getOriginTable().getRowType().getFieldList().get(childs.get(0).getCluster().getMetadataQuery().getColumnOrigin(childs.get(0),index).getOriginColumnOrdinal()).getType();
            }
        }
        else {
            index-=childs.get(0).getRowType().getFieldNames().size();
            if(childs.get(1).getCluster().getMetadataQuery().getColumnOrigin(childs.get(1), index) != null){
                res += childs.get(1).getCluster().getMetadataQuery().getColumnOrigin(childs.get(1),index).getOriginTable().getQualifiedName().get(0);
                res += "."+childs.get(1).getCluster().getMetadataQuery().getColumnOrigin(childs.get(1),index).getOriginTable().getRowType().getFieldNames().get(childs.get(1).getCluster().getMetadataQuery().getColumnOrigin(childs.get(1),index).getOriginColumnOrdinal());
                res += "."+childs.get(1).getCluster().getMetadataQuery().getColumnOrigin(childs.get(1),index).getOriginTable().getRowType().getFieldList().get(childs.get(1).getCluster().getMetadataQuery().getColumnOrigin(childs.get(1),index).getOriginColumnOrdinal()).getType();
            }
        }
        return res;
    }
    public static void dfs_mtcs_tree(Node node, int depth){
        if(node.parent == null){
            System.out.println("Original Query");
            System.out.println(node.state_rel.explain());
            // System.out.println(node.activatedRules);
            return;
        }
        dfs_mtcs_tree(node.parent, depth + 1);
        System.out.println(node.activatedRules);
        System.out.println(node.state_rel.explain());
    }
}
