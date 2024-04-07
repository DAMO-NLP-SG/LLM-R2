package main;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import main.trait.SimpleTable;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.Frameworks;
import org.apache.commons.lang3.tuple.Pair;

import java.util.Map;
import java.util.Vector;


public class GenerateSchema {
    public static SchemaPlus generate_schema(JSONArray json, Vector<Pair<String, Vector<Pair<String, String>>>>schema_all) {
        try {
            SchemaPlus rootSchema = Frameworks.createRootSchema(true);
            for (int i = 0; i < json.size(); i++) {
                JSONObject object = json.getJSONObject(i);
                SimpleTable.Builder temTable = SimpleTable.newBuilder(object.getString("table"));
                JSONArray columnsJsonArray = object.getJSONArray("columns");

                for (int j = 0; j < columnsJsonArray.size(); j++) {
                    JSONObject subJson = columnsJsonArray.getJSONObject(j);
                    String type = subJson.get("type").toString();

                    // 防止类型获取失败
                    if (SqlTypeName.get(type.toUpperCase()) == null) {
                        if (type.equalsIgnoreCase("character varying")) {
                            type = "varchar";
                        } else if (type.equalsIgnoreCase("character")) {
                            type = "char";
                        } else if (type.equalsIgnoreCase("numeric")) {
                            type = "decimal";
                        } else if (type.equalsIgnoreCase("string")) {
                            type = "varchar";
                        } else if (type.equalsIgnoreCase("int")) {
                            type = "integer";
                        } else if (type.toLowerCase().contains("decimal")) {
                            type = "decimal";
                        } else {
                            System.out.println("类型转换失败，临时转成varchar, 字段为:" + subJson.get("name").toString() + "，类型为:" + type);
                            type = "varchar";
                        }
                    }
                    SqlTypeName typeName = SqlTypeName.get(type.toUpperCase());
                    temTable.addField((String) subJson.get("name"), typeName);
                }
                temTable.withRowCount(object.getInteger("rows"));
                Pair<String, Vector<Pair<String, String>>> v = temTable.tableDiscribe();
                schema_all.add(v);
                rootSchema.add(object.getString("table"), temTable.build());
            }
            return rootSchema;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
